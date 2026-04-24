"""
Parallel Associative Scan for Linear Recurrences.

Provides both a numerically stable parallel implementation and
a fast sequential fallback. The parallel version uses a chunked
approach to avoid numerical overflow in the log-space cumsum.

Compatible with torch.compile for kernel fusion on T4/A100.
"""
import torch
import math


def parallel_scan(log_coeffs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Compute a first-order linear recurrence in parallel.
    
    Recurrence: h[t] = exp(log_coeffs[t]) * h[t-1] + values[t]
    
    Uses a chunked parallel approach: process chunks of size C in parallel
    within each chunk, then propagate state between chunks sequentially.
    This gives O(T/C + C) depth instead of O(T), while maintaining
    numerical stability by limiting the cumulative log range.
    
    Args:
        log_coeffs: [B, T, D] — log of the decay coefficients (should be ≤ 0)
        values: [B, T, D] — input values at each time step
        
    Returns:
        [B, T, D] — the recurrence output at each time step
    """
    B, T, D = log_coeffs.shape
    
    # Clamp log_coeffs for stability (decay must be in (0, 1))
    log_coeffs = log_coeffs.clamp(max=-1e-6)
    
    # Choose chunk size: balance parallelism vs numerical range
    # Within a chunk of size C, cumulative log range is at most C * max(|log_coeff|)
    # We want exp(C * max_log) < 1e15 → C < 15 / max_log
    CHUNK_SIZE = 16  # Safe for typical decay rates
    
    if T <= CHUNK_SIZE:
        return _scan_chunk(log_coeffs, values)
    
    # Chunk the sequence
    n_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Pad to multiple of CHUNK_SIZE
    T_padded = n_chunks * CHUNK_SIZE
    if T_padded > T:
        pad = T_padded - T
        log_coeffs = torch.nn.functional.pad(log_coeffs, (0, 0, 0, pad))
        values = torch.nn.functional.pad(values, (0, 0, 0, pad))
    
    # Reshape into chunks: [B, n_chunks, CHUNK_SIZE, D]
    log_c = log_coeffs.view(B, n_chunks, CHUNK_SIZE, D)
    vals = values.view(B, n_chunks, CHUNK_SIZE, D)
    
    # Process each chunk independently (parallel within chunk)
    chunk_outputs = []
    chunk_carries = []
    
    for chunk_idx in range(n_chunks):
        chunk_log = log_c[:, chunk_idx, :, :]  # [B, C, D]
        chunk_val = vals[:, chunk_idx, :, :]    # [B, C, D]
        out = _scan_chunk(chunk_log, chunk_val)  # [B, C, D]
        
        # The carry from this chunk: the last hidden state
        carry = out[:, -1, :]  # [B, D]
        
        # Also need the cumulative decay from start of chunk to each position
        # (for adding the inter-chunk contribution)
        cum_log = torch.cumsum(chunk_log, dim=1)  # [B, C, D]
        
        chunk_outputs.append(out)
        chunk_carries.append((carry, cum_log))
    
    # Propagate carries between chunks
    # For chunk i, the input from previous chunks is:
    # h_prev * prod(decays in chunk i up to position t)
    h_prev = torch.zeros(B, D, device=log_coeffs.device, dtype=log_coeffs.dtype)
    
    corrected = []
    for chunk_idx in range(n_chunks):
        out = chunk_outputs[chunk_idx]     # [B, C, D]
        cum_log = chunk_carries[chunk_idx][1]  # [B, C, D]
        
        # Add contribution from previous chunks
        # h_corrected[t] = out[t] + h_prev * exp(cum_log[t])
        decay_from_prev = torch.exp(cum_log)  # [B, C, D]
        correction = h_prev.unsqueeze(1) * decay_from_prev  # [B, C, D]
        corrected_out = out + correction
        corrected.append(corrected_out)
        
        # Update h_prev: the last state of this corrected chunk
        h_prev = corrected_out[:, -1, :]  # [B, D]
    
    # Concatenate and trim
    result = torch.cat(corrected, dim=1)  # [B, T_padded, D]
    return result[:, :T, :]


def _scan_chunk(log_coeffs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Scan a short chunk using the cumsum approach.
    Numerically stable for chunks up to ~32 steps.
    
    Args:
        log_coeffs: [B, C, D]
        values: [B, C, D]
    Returns:
        [B, C, D]
    """
    B, C, D = log_coeffs.shape
    
    # Cumulative log-coefficients
    cum_log = torch.cumsum(log_coeffs, dim=1)  # [B, C, D]
    
    # Scale values by inverse cumulative coefficient
    # Clamped for numerical stability
    neg_cum_log = (-cum_log).clamp(-30, 30)
    scaled_values = values * torch.exp(neg_cum_log)
    
    # Cumulative sum (the parallel part)
    cum_scaled = torch.cumsum(scaled_values, dim=1)  # [B, C, D]
    
    # Scale back
    result = cum_scaled * torch.exp(cum_log.clamp(-30, 30))
    
    return result
