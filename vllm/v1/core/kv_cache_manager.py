# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import time
from typing import Dict

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus, StreamingRequest, StreamingPolicy

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """
    blocks: tuple[list[KVCacheBlock], ...]
    """
    blocks[i][j] refers to the i-th kv_cache_group and the j-th block of tokens.
    We don't use block of tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but 
    will be broken if we want to give different block_size to different 
    kv_cache_groups in the future.
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(blk1 + blk2
                  for blk1, blk2 in zip(self.blocks, other.blocks)))

    def get_block_ids(self) -> tuple[list[int], ...]:
        """
        Converts the KVCacheBlocks instance to block_ids.
        
        Returns:
            tuple[list[int], ...]: A tuple of lists where
            * the outer tuple corresponds to KV cache groups
            * each inner list contains the block_ids of the blocks in that group
        """
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id for block in self.blocks[0]
            if block.block_hash is None
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size: Optional[int] = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted 
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return KVCacheBlocks(
            self.coordinator.get_blocks(request_id)).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))


@dataclass
class SessionCacheState:
    """State tracking for streaming session KV cache with sequence management."""
    session_id: str
    kv_cache_blocks: Optional[KVCacheBlocks] = None
    last_frame_sequence: int = 0
    context_length: int = 0
    max_tokens: int = 0
    allocated_token_budget: float = 1.0  # Default 100% budget
    effective_max_tokens: int = 0
    session_start_time: float = 0
    last_access_time: float = 0
    
    # Sequence tracking inspired by MiniCPM-o
    last_completed_sequence: int = 0  # Last fully processed sequence
    processing_sequences: set = None  # Currently processing sequences
    max_context_capacity: int = 8192   # Maximum context before reset
    
    def __post_init__(self):
        if self.session_start_time == 0:
            self.session_start_time = time.time()
        if self.last_access_time == 0:
            self.last_access_time = time.time()
        if self.effective_max_tokens == 0:
            self.effective_max_tokens = int(self.max_tokens * self.allocated_token_budget)
        if self.processing_sequences is None:
            self.processing_sequences = set()
    
    def update_token_budget(self, budget_share: float):
        """Update token budget allocation based on streaming policy."""
        self.allocated_token_budget = budget_share
        self.effective_max_tokens = int(self.max_tokens * budget_share)
        self.last_access_time = time.time()
        
    def reset_kv_cache(self):
        """Reset KV cache for new session."""
        self.kv_cache_blocks = None
        self.context_length = 0
        self.last_completed_sequence = 0
        self.processing_sequences.clear()
        
    def start_processing_sequence(self, seq_id: int):
        """Mark sequence as starting processing."""
        self.processing_sequences.add(seq_id)
        logger.debug(f"Session {self.session_id} started processing sequence {seq_id}")
        
    def complete_sequence(self, seq_id: int):
        """Mark sequence as completed."""
        self.processing_sequences.discard(seq_id)
        self.last_completed_sequence = max(self.last_completed_sequence, seq_id)
        logger.debug(f"Session {self.session_id} completed sequence {seq_id}")
        
    def check_capacity_overflow(self, additional_tokens: int = 0) -> bool:
        """Check if adding more tokens would cause capacity overflow inspired by MiniCPM-o."""
        projected_length = self.context_length + additional_tokens
        return projected_length >= self.max_context_capacity * 0.9  # 90% threshold
        
    def should_reset_cache(self) -> bool:
        """Determine if cache should be reset due to capacity issues."""
        return (self.context_length >= self.max_context_capacity or 
                self.context_length >= self.effective_max_tokens)
                
    def touch(self):
        """Update last access time for LRU tracking."""
        self.last_access_time = time.time()


class SessionAwareKVCacheManager(KVCacheManager):
    """Extended KV Cache Manager with support for streaming session management and policy-based budgeting."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Session-level cache management
        self.session_caches: Dict[str, SessionCacheState] = {}
        self.session_cleanup_threshold = 3600  # Cleanup sessions inactive for 1 hour
        self.last_cleanup_time = time.time()
        
        logger.info("Initialized SessionAwareKVCacheManager with session-level cache management")
        
    def get_session_cache(self, session_id: str, max_tokens: Optional[int] = None) -> SessionCacheState:
        """Get or create session-level KV cache state.
        
        Args:
            session_id: Unique session identifier
            max_tokens: Maximum tokens for this session (defaults to model max)
            
        Returns:
            SessionCacheState for the given session
        """
        if session_id not in self.session_caches:
            self.session_caches[session_id] = SessionCacheState(
                session_id=session_id,
                max_tokens=max_tokens or self.max_model_len,
                session_start_time=time.time()
            )
            logger.debug(f"Created new session cache: {session_id}")
            
        session_cache = self.session_caches[session_id]
        session_cache.touch()
        return session_cache
        
    def append_kv_cache_with_policy(self, 
                                  session_id: str,
                                  new_kv_data: KVCacheBlocks,
                                  policy: StreamingPolicy,
                                  sequence_id: int = None) -> KVCacheBlocks:
        """Append KV cache data to a session with policy constraints and capacity management.
        
        Args:
            session_id: Session identifier
            new_kv_data: New KV cache blocks to append
            policy: Streaming policy with budget constraints
            sequence_id: Sequence ID for tracking
            
        Returns:
            Updated KV cache blocks for the session
        """
        session_cache = self.get_session_cache(session_id)
        
        # Apply token budget policy
        if policy.token_budget_share is not None:
            session_cache.update_token_budget(policy.token_budget_share)
        
        # Mark sequence as processing if provided
        if sequence_id is not None:
            session_cache.start_processing_sequence(sequence_id)
            
        # Check capacity overflow inspired by MiniCPM-o
        new_tokens = self._estimate_kv_length(new_kv_data)
        
        if session_cache.check_capacity_overflow(new_tokens):
            logger.warning(f"Session {session_id} approaching capacity limit, "
                          f"context_length={session_cache.context_length}, "
                          f"new_tokens={new_tokens}, "
                          f"capacity={session_cache.max_context_capacity}")
            
            # Apply reset strategy if necessary
            if session_cache.should_reset_cache():
                logger.info(f"Resetting KV cache for session {session_id} due to capacity overflow")
                session_cache.reset_kv_cache()
                
        projected_length = session_cache.context_length + new_tokens
        
        # Check token budget constraints
        if projected_length > session_cache.effective_max_tokens:
            logger.warning(f"Session {session_id} would exceed token budget "
                          f"({projected_length} > {session_cache.effective_max_tokens}), "
                          f"applying eviction policy")
            session_cache = self._evict_cache_blocks(session_cache, projected_length)
            
        # Append new KV data
        if session_cache.kv_cache_blocks is None:
            session_cache.kv_cache_blocks = new_kv_data
        else:
            session_cache.kv_cache_blocks = session_cache.kv_cache_blocks + new_kv_data
            
        # Update context length and mark sequence as completed
        session_cache.context_length = projected_length
        if sequence_id is not None:
            session_cache.complete_sequence(sequence_id)
        session_cache.touch()
        
        return session_cache.kv_cache_blocks
        
    def get_session_computed_blocks(self, 
                                  session_id: str,
                                  request: StreamingRequest) -> tuple[KVCacheBlocks, int]:
        """Get computed blocks for a streaming session with incremental processing.
        
        Args:
            session_id: Session identifier
            request: Streaming request
            
        Returns:
            Tuple of (computed blocks, number of computed tokens)
        """
        session_cache = self.get_session_cache(session_id)
        
        # For streaming requests, we can reuse session-level cached blocks
        if session_cache.kv_cache_blocks is not None and not request.is_first_chunk:
            # Return existing blocks for incremental processing
            return session_cache.kv_cache_blocks, session_cache.context_length
        else:
            # First chunk or no cache - use standard prefix cache logic
            return self.get_computed_blocks(request)
            
    def allocate_session_slots(self,
                             session_id: str, 
                             request: StreamingRequest,
                             num_new_tokens: int,
                             num_computed_tokens: int = 0) -> KVCacheBlocks:
        """Allocate KV cache slots for a streaming session.
        
        Args:
            session_id: Session identifier
            request: Streaming request
            num_new_tokens: Number of new tokens to allocate
            num_computed_tokens: Number of already computed tokens
            
        Returns:
            Allocated KV cache blocks
        """
        session_cache = self.get_session_cache(session_id)
        policy = request.get_policy()
        
        # Apply token budget constraints
        if policy.token_budget_share is not None:
            session_cache.update_token_budget(policy.token_budget_share)
            
        # Check budget before allocation
        total_tokens = session_cache.context_length + num_new_tokens
        if total_tokens > session_cache.effective_max_tokens:
            # Reduce allocation to fit budget
            available_tokens = max(0, session_cache.effective_max_tokens - session_cache.context_length)
            num_new_tokens = min(num_new_tokens, available_tokens)
            logger.debug(f"Reduced allocation for session {session_id} due to budget: "
                        f"{num_new_tokens} tokens")
            
        if num_new_tokens <= 0:
            return self.create_empty_block_list()
            
        # Use standard allocation logic
        allocated_blocks = self.allocate_slots(request, num_new_tokens, num_computed_tokens)
        
        # Update session state
        session_cache.context_length += num_new_tokens
        session_cache.touch()
        
        return allocated_blocks
        
    def _estimate_kv_length(self, kv_blocks: KVCacheBlocks) -> int:
        """Estimate the number of tokens represented by KV cache blocks."""
        if not kv_blocks.blocks or not kv_blocks.blocks[0]:
            return 0
        # Rough estimate: block_size * number of blocks
        return len(kv_blocks.blocks[0]) * (self.block_size or 16)
        
    def _evict_cache_blocks(self, 
                          session_cache: SessionCacheState,
                          target_length: int) -> SessionCacheState:
        """Evict cache blocks when session exceeds token budget.
        
        Implements a simple strategy: keep most recent blocks up to budget.
        """
        if session_cache.kv_cache_blocks is None:
            return session_cache
            
        # Calculate how many blocks to keep
        blocks_to_keep = session_cache.effective_max_tokens // (self.block_size or 16)
        
        if blocks_to_keep <= 0:
            # No budget - clear all blocks
            session_cache.reset_kv_cache()
            logger.info(f"Cleared all cache blocks for session {session_cache.session_id} "
                       f"due to zero token budget")
        else:
            # Keep most recent blocks
            for group_idx, group in enumerate(session_cache.kv_cache_blocks.blocks):
                if len(group) > blocks_to_keep:
                    # Keep last N blocks
                    session_cache.kv_cache_blocks.blocks[group_idx] = group[-blocks_to_keep:]
                    
            # Update context length
            session_cache.context_length = min(
                session_cache.context_length, 
                session_cache.effective_max_tokens
            )
            
            logger.info(f"Evicted cache blocks for session {session_cache.session_id}: "
                       f"kept {blocks_to_keep} blocks, "
                       f"context_length={session_cache.context_length}")
                       
        return session_cache
        
    def remove_session(self, session_id: str) -> None:
        """Remove a streaming session and cleanup its cache resources."""
        if session_id in self.session_caches:
            session_cache = self.session_caches[session_id]
            duration = time.time() - session_cache.session_start_time
            
            # Cleanup any allocated blocks
            if session_cache.kv_cache_blocks is not None:
                # Note: In a full implementation, we'd need to return blocks to pool
                pass
                
            del self.session_caches[session_id]
            logger.info(f"Removed session cache {session_id}: "
                       f"duration={duration:.2f}s, "
                       f"context_length={session_cache.context_length}")
                       
    def cleanup_inactive_sessions(self, max_age_seconds: Optional[int] = None) -> int:
        """Cleanup sessions that have been inactive for too long.
        
        Args:
            max_age_seconds: Maximum age for inactive sessions (default: class threshold)
            
        Returns:
            Number of sessions cleaned up
        """
        max_age = max_age_seconds or self.session_cleanup_threshold
        current_time = time.time()
        
        # Only run cleanup periodically
        if current_time - self.last_cleanup_time < 300:  # 5 minutes
            return 0
            
        inactive_sessions = []
        for session_id, session_cache in self.session_caches.items():
            age = current_time - session_cache.last_access_time
            if age > max_age:
                inactive_sessions.append(session_id)
                
        # Remove inactive sessions
        for session_id in inactive_sessions:
            self.remove_session(session_id)
            
        self.last_cleanup_time = current_time
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
            
        return len(inactive_sessions)
        
    def get_session_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics for all active sessions."""
        stats = {}
        current_time = time.time()
        
        for session_id, session_cache in self.session_caches.items():
            stats[session_id] = {
                'context_length': session_cache.context_length,
                'max_tokens': session_cache.max_tokens,
                'effective_max_tokens': session_cache.effective_max_tokens,
                'token_budget_share': session_cache.allocated_token_budget,
                'duration': current_time - session_cache.session_start_time,
                'inactive_time': current_time - session_cache.last_access_time,
                'last_frame_sequence': session_cache.last_frame_sequence
            }
            
        return stats
        
    def list_active_sessions(self) -> list[str]:
        """List all active streaming session IDs."""
        return list(self.session_caches.keys())
