# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.request import StreamingRequest, StreamingPolicy
from vllm.v1.utils import is_streaming_input_enabled

logger = init_logger(__name__)


@dataclass 
class ProcessedChunk:
    """Represents a processed chunk of multimodal data with TDM metadata."""
    modality: str  # 'video', 'audio', 'text'
    data: Any
    timestamp: float
    time_slice: int
    sequence_id: int = 0


class StreamingProcessor(Processor):
    """Extended processor with TDM (Time Division Multiplexing) and streaming policy support.
    
    This processor handles streaming multimodal input by:
    1. Implementing TDM to serialize parallel multimodal streams
    2. Applying streaming policies (latency budget, vision stride, token budget)
    3. Converting incremental frame data to token sequences
    4. Maintaining compatibility with base Processor interface for polymorphism
    """
    
    def __init__(self, 
                 vllm_config: VllmConfig,
                 tokenizer: TokenizerGroup,
                 mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
                 time_slice_ms: int = 100):
        super().__init__(vllm_config, tokenizer, mm_registry)
        
        # TDM configuration
        self.time_slice_ms = time_slice_ms
        self.modality_queues: Dict[str, deque] = {
            'video': deque(),
            'audio': deque(), 
            'text': deque()
        }
        
        # Session contexts for incremental processing
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Modality processing priority
        self.modality_priority = {'text': 0, 'video': 1, 'audio': 2}
        
        logger.info(f"Initialized StreamingProcessor with {time_slice_ms}ms time slices")
    
    def process_inputs(self, *args, **kwargs) -> tuple[Optional[str], EngineCoreRequest]:
        """Override base process_inputs to handle both regular and streaming requests.
        
        Maintains polymorphic compatibility - can handle regular requests via super(),
        and streaming requests via process_streaming_multimodal().
        """
        # Check if streaming input is enabled
        if not is_streaming_input_enabled():
            return super().process_inputs(*args, **kwargs)
            
        # Check if this is a streaming request by looking for StreamingRequest type
        request_id = args[0] if args else kwargs.get('request_id')
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt')
        
        # If prompt is a StreamingRequest, handle via streaming path
        if isinstance(prompt, StreamingRequest):
            streaming_request = prompt
            core_request = self.process_streaming_multimodal(streaming_request)
            return None, core_request  # Return format matching base class
        
        # Otherwise, delegate to parent class for standard processing
        return super().process_inputs(*args, **kwargs)
        
    def process_streaming_multimodal(self, request: StreamingRequest) -> EngineCoreRequest:
        """Process a streaming multimodal request with TDM and policy support.
        
        Args:
            request: StreamingRequest containing frame data and policy
            
        Returns:
            EngineCoreRequest ready for engine core processing
        """
        # Get session context
        session_context = self._get_session_context(request.session_id)
        
        # Get current policy directly from request (no external manager)
        policy = request.get_policy()
        
        # Get current frame data
        frame_data = request.get_current_frame()
        if not frame_data:
            logger.warning(f"No frame data for request {request.request_id}")
            return self._create_empty_core_request(request)
            
        # Apply TDM: multiplex modalities into time slices
        tdm_chunks = self._multiplex_modalities(frame_data, request.frame_sequence_id)
        
        # Apply streaming policies
        if policy.vision_stride and policy.vision_stride > 1:
            tdm_chunks = self._apply_vision_stride(tdm_chunks, policy.vision_stride)
            
        # Check latency budget
        if policy.latency_budget:
            tdm_chunks = self._apply_latency_budget(tdm_chunks, policy.latency_budget)
            
        # Convert to incremental token sequence
        incremental_tokens = self._tokenize_incremental_input(
            tdm_chunks, session_context, policy
        )
        
        # Process multimodal placeholders
        mm_kwargs, mm_placeholders, mm_hashes = self._process_multimodal_data(
            tdm_chunks, request
        )
        
        # Create core request
        core_request = EngineCoreRequest(
            request_id=request.request_id,
            prompt_token_ids=incremental_tokens,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority
        )
        
        # Update session context with new tokens
        self._update_session_context(request.session_id, incremental_tokens, policy)
        
        logger.debug(f"Processed streaming request {request.request_id}: "
                    f"{len(incremental_tokens)} tokens, "
                    f"{len(tdm_chunks)} TDM chunks")
        
        return core_request
        
    def _multiplex_modalities(self, frame_data: MultiModalKwargsItem, 
                             sequence_id: int) -> List[ProcessedChunk]:
        """TDM Core Logic: Multiplex multimodal data into time-ordered chunks.
        
        This implements the Time Division Multiplexing mechanism from MiniCPM-o:
        - Divides parallel modality streams into sequential info within time slices
        - Each modality gets assigned to specific time slices
        - Maintains temporal coherence across modalities
        """
        chunks = []
        current_time = time.time()
        base_time_slice = int(current_time * 1000 / self.time_slice_ms)
        
        # Process visual data
        if hasattr(frame_data, 'data') and 'image' in str(type(frame_data.data)):
            chunks.append(ProcessedChunk(
                modality='video',
                data=frame_data.data,
                timestamp=current_time,
                time_slice=base_time_slice,
                sequence_id=sequence_id
            ))
            
        # Process audio data 
        if hasattr(frame_data, 'data') and 'audio' in str(type(frame_data.data)):
            # Split audio into smaller time slices if needed
            audio_chunks = self._split_audio_by_timeslice(
                frame_data.data, current_time, base_time_slice, sequence_id
            )
            chunks.extend(audio_chunks)
            
        # Process text data
        if hasattr(frame_data, 'data') and isinstance(frame_data.data, str):
            chunks.append(ProcessedChunk(
                modality='text',
                data=frame_data.data,
                timestamp=current_time,
                time_slice=base_time_slice,
                sequence_id=sequence_id
            ))
            
        # Serialize chunks using TDM
        return self._serialize_chunks_tdm(chunks)
        
    def _serialize_chunks_tdm(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """TDM Serialization: Convert parallel streams to sequential time slices.
        
        Key insight from MiniCPM-o: Transform concurrent multimodal input into
        sequential processing within small periodic time slices.
        """
        if not chunks:
            return []
            
        # Sort by time slice first, then by modality priority
        chunks.sort(key=lambda x: (x.time_slice, self.modality_priority.get(x.modality, 999)))
        
        serialized = []
        current_slice = None
        slice_chunks = []
        
        for chunk in chunks:
            if current_slice != chunk.time_slice:
                # Process completed time slice
                if slice_chunks:
                    # Within each time slice, maintain modality priority
                    slice_chunks.sort(key=lambda x: self.modality_priority.get(x.modality, 999))
                    serialized.extend(slice_chunks)
                    
                # Start new time slice
                current_slice = chunk.time_slice
                slice_chunks = [chunk]
            else:
                slice_chunks.append(chunk)
                
        # Process final time slice
        if slice_chunks:
            slice_chunks.sort(key=lambda x: self.modality_priority.get(x.modality, 999))
            serialized.extend(slice_chunks)
            
        logger.debug(f"TDM serialized {len(chunks)} chunks across "
                    f"{len(set(c.time_slice for c in chunks))} time slices")
        return serialized
        
    def _apply_vision_stride(self, chunks: List[ProcessedChunk], 
                           stride: int) -> List[ProcessedChunk]:
        """Apply vision stride policy: process every N video frames."""
        filtered_chunks = []
        video_frame_count = 0
        
        for chunk in chunks:
            if chunk.modality == 'video':
                # Only keep every stride-th video frame
                if video_frame_count % stride == 0:
                    filtered_chunks.append(chunk)
                video_frame_count += 1
            else:
                # Keep all non-video chunks
                filtered_chunks.append(chunk)
                
        logger.debug(f"Vision stride {stride}: kept {len(filtered_chunks)}/{len(chunks)} chunks")
        return filtered_chunks
        
    def _apply_latency_budget(self, chunks: List[ProcessedChunk], 
                            budget_ms: float) -> List[ProcessedChunk]:
        """Apply latency budget: filter chunks that exceed processing budget."""
        current_time_ms = time.time() * 1000
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_age_ms = current_time_ms - (chunk.timestamp * 1000)
            if chunk_age_ms <= budget_ms:
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Dropped chunk due to latency budget: "
                           f"{chunk_age_ms:.1f}ms > {budget_ms}ms")
                           
        return filtered_chunks
        
    def _tokenize_incremental_input(self, chunks: List[ProcessedChunk],
                                   session_context: Dict[str, Any],
                                   policy: StreamingPolicy) -> List[int]:
        """Convert TDM chunks to incremental token sequence.
        
        This follows MiniCPM-o's approach of converting time-serialized
        multimodal chunks into a coherent token sequence.
        """
        if not chunks:
            return []
            
        # Build incremental prompt from TDM chunks
        prompt_parts = []
        
        for chunk in chunks:
            if chunk.modality == 'text':
                prompt_parts.append(str(chunk.data))
            elif chunk.modality == 'video':
                prompt_parts.append("(<video>./</video>)")
            elif chunk.modality == 'audio':
                # Audio not supported in current implementation
                continue
                
        # Combine with session context
        if session_context.get('accumulated_prompt'):
            full_prompt = session_context['accumulated_prompt'] + " " + " ".join(prompt_parts)
        else:
            full_prompt = " ".join(prompt_parts)
            
        # Apply token budget policy
        if policy.token_budget_share and policy.token_budget_share < 1.0:
            # Estimate max tokens and truncate if needed
            estimated_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
            max_allowed = self.model_config.max_model_len * policy.token_budget_share
            if estimated_tokens > max_allowed:
                words = full_prompt.split()
                truncated_words = words[:int(max_allowed / 1.3)]
                full_prompt = " ".join(truncated_words)
                logger.debug(f"Truncated prompt due to token budget: "
                           f"{len(words)} -> {len(truncated_words)} words")
                
        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer.encode(full_prompt)
        else:
            # Fallback for when tokenizer is not available
            tokens = list(range(len(full_prompt.split())))
            
        return tokens
        
    def _split_audio_by_timeslice(self, audio_data: Any, base_timestamp: float,
                                base_time_slice: int, sequence_id: int) -> List[ProcessedChunk]:
        """Split audio data across multiple time slices for TDM."""
        # This is a simplified implementation - in practice would need
        # to handle actual audio data format and splitting
        chunks = []
        
        # For now, treat as single chunk
        chunks.append(ProcessedChunk(
            modality='audio',
            data=audio_data,
            timestamp=base_timestamp,
            time_slice=base_time_slice + 1,  # Offset audio by 1 slice
            sequence_id=sequence_id
        ))
        
        return chunks
        
    def _process_multimodal_data(self, chunks: List[ProcessedChunk], 
                               request: StreamingRequest):
        """Process multimodal data and create placeholders."""
        mm_kwargs = []
        mm_placeholders = []
        mm_hashes = []
        
        for chunk in chunks:
            if chunk.modality in ['video', 'audio']:
                # Create multimodal kwargs for non-text modalities
                mm_kwargs.append(MultiModalKwargsItem(
                    data=chunk.data,
                    modality=chunk.modality
                ))
                # Note: Placeholder and hash generation would need proper implementation
                
        return mm_kwargs, mm_placeholders, mm_hashes
        
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context for incremental processing."""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                'accumulated_prompt': '',
                'total_tokens': 0,
                'frame_count': 0,
                'start_time': time.time()
            }
        return self.session_contexts[session_id]
        
    def _update_session_context(self, session_id: str, tokens: List[int],
                              policy: StreamingPolicy) -> None:
        """Update session context with new processing results."""
        context = self.session_contexts[session_id]
        context['total_tokens'] += len(tokens)
        context['frame_count'] += 1
        
        # Apply token budget cleanup if needed
        if (policy.token_budget_share and 
            context['total_tokens'] > self.model_config.max_model_len * policy.token_budget_share):
            # Could implement context cleanup/eviction here
            logger.debug(f"Session {session_id} approaching token budget limit")
            
    def _create_empty_core_request(self, request: StreamingRequest) -> EngineCoreRequest:
        """Create empty core request for cases with no frame data."""
        return EngineCoreRequest(
            request_id=request.request_id,
            prompt_token_ids=[],
            mm_kwargs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request
        )
        
    def cleanup_session(self, session_id: str) -> None:
        """Cleanup session context when streaming session ends."""
        if session_id in self.session_contexts:
            context = self.session_contexts[session_id]
            duration = time.time() - context['start_time']
            logger.info(f"Cleaned up session {session_id}: "
                       f"duration={duration:.2f}s, "
                       f"frames={context['frame_count']}, "
                       f"total_tokens={context['total_tokens']}")
            del self.session_contexts[session_id]
            
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a streaming session."""
        context = self.session_contexts.get(session_id)
        if context:
            return {
                'session_id': session_id,
                'duration': time.time() - context['start_time'],
                'frame_count': context['frame_count'],
                'total_tokens': context['total_tokens'],
                'avg_tokens_per_frame': context['total_tokens'] / max(1, context['frame_count'])
            }
        return None
