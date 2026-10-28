import unittest, numpy as np
from engine.dequant import block_layout, dequantize


def _q8_activation_reference(raw, ggml_type, rows, cols, x):
 # Match the native Q4_K/Q5_K kernel's int8 activation quantization while
 # retaining the exact float activation sum for each affine minimum term.
 bytes_per_block = 144 if ggml_type == 12 else 176
 blocks = cols // 256
 quant_groups = cols // 32
 sum_delta = np.empty(quant_groups, dtype=np.float32)
 x_approx = np.empty(cols, dtype=np.float32)
 for group in range(quant_groups):
  first = group * 32
  values = x[first:first + 32]
  scale = max(float(np.max(np.abs(values))), 1e-20) / 127.0
  q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
  x_approx[first:first + 32] = q.astype(np.float32) * scale
  sum_delta[group] = float(values.sum() - x_approx[first:first + 32].sum())
 expected = dequantize(raw, ggml_type, (rows, cols)) @ x_approx
 view = memoryview(raw)
 for row in range(rows):
  for block_idx in range(blocks):
   offset = (row * blocks + block_idx) * bytes_per_block
   dm = float(np.frombuffer(view[offset + 2:offset + 4], dtype='<f2')[0])
   scales = view[offset + 4:offset + 16]
   for group in range(8):
    if group < 4:
     minimum_q = scales[group + 4] & 0x3f
    else:
     minimum_q = (scales[group + 4] >> 4) | ((scales[group - 4] >> 6) << 4)
    expected[row] -= dm * minimum_q * sum_delta[block_idx * 8 + group]
 return expected

class DequantTests(unittest.TestCase):
 def test_layouts(self):
  self.assertEqual(block_layout(0),(1,4)); self.assertEqual(block_layout(1),(1,2)); self.assertEqual(block_layout(2),(32,18)); self.assertEqual(block_layout(3),(32,20)); self.assertEqual(block_layout(6),(32,22)); self.assertEqual(block_layout(7),(32,24)); self.assertEqual(block_layout(8),(32,34)); self.assertEqual(block_layout(12),(256,144)); self.assertEqual(block_layout(13),(256,176)); self.assertEqual(block_layout(14),(256,210))
 def test_unsupported_is_error(self):
  with self.assertRaises(NotImplementedError):block_layout(99)
 def test_q8(self):
  raw=np.float16(0.5).tobytes()+np.arange(-16,16,dtype=np.int8).tobytes(); out=dequantize(raw,8,(32,)); np.testing.assert_allclose(out,np.arange(-16,16,dtype=np.float32)*.5)
 def test_q5_0_high_bits_and_signed_range(self):
  raw=np.float16(1).tobytes()+bytes(4)+bytes([0x98])*16
  out=dequantize(raw,6,(32,))
  np.testing.assert_allclose(out[:16],-8); np.testing.assert_allclose(out[16:],-7)
 def test_q5_1_affine_zero_point(self):
  raw=np.float16(.5).tobytes()+np.float16(1).tobytes()+bytes(4)+bytes([0x21])*16
  out=dequantize(raw,7,(32,))
  np.testing.assert_allclose(out[:16],1.5); np.testing.assert_allclose(out[16:],2.0)
 def test_unquantized_float_tensors(self):
  values=np.array([1.25,-2.5,0.0],dtype=np.float32)
  np.testing.assert_array_equal(dequantize(values.tobytes(),0,(3,)),values)
  np.testing.assert_allclose(dequantize(values.astype(np.float16).tobytes(),1,(3,)),values)
 def test_q4_0_nibble_order(self):
  raw=np.float16(1).tobytes()+bytes([0x98])*16
  out=dequantize(raw,2,(32,))
  np.testing.assert_allclose(out[:16],0); np.testing.assert_allclose(out[16:],1)
 def test_q4_1_affine_scale_and_minimum(self):
  raw=np.float16(.5).tobytes()+np.float16(1).tobytes()+bytes([0x21])*16
  out=dequantize(raw,3,(32,))
  np.testing.assert_allclose(out[:16],1.5); np.testing.assert_allclose(out[16:],2.0)
 def test_q4_k_interleaves_low_and_high_nibble_chunks(self):
  # Each 32-byte qs chunk yields one low-nibble group and then one high-nibble group.
  scales=bytes([1,1,1,1,0,0,0,0,1,1,1,1])
  qs=bytes([0x10]*32+[0x20]*32+[0x30]*32+[0x40]*32)
  raw=np.float16(1).tobytes()+np.float16(0).tobytes()+scales+qs
  out=dequantize(raw,12,(256,))
  for group, expected in enumerate((0,1,0,2,0,3,0,4)):
   np.testing.assert_allclose(out[group*32:(group+1)*32], expected)

 def test_q4_k_zero_block(self):
  raw=np.float16(1).tobytes()+np.float16(0).tobytes()+bytes(12+128)
  np.testing.assert_allclose(dequantize(raw,12,(256,)),0)
 def test_q5_k_uses_high_bit_planes_and_q4_k_scale_layout(self):
  scales=bytes([1,1,1,1,0,0,0,0,1,1,1,1])
  qs=bytes([0x10]*32+[0x20]*32+[0x30]*32+[0x40]*32)
  raw=np.float16(1).tobytes()+np.float16(0).tobytes()+scales+bytes([0xFF])*32+qs
  out=dequantize(raw,13,(256,))
  for group, expected in enumerate((16,17,16,18,16,19,16,20)):
   np.testing.assert_allclose(out[group*32:(group+1)*32], expected)
 def test_q4_k_nibble_order_and_scale(self):
  scales=bytes([1,1,1,1,0,0,0,0,1,1,1,1]); qs=bytes([0x21])*128
  raw=np.float16(1).tobytes()+np.float16(0).tobytes()+scales+qs
  out=dequantize(raw,12,(256,))
  for group in range(8):
   expected = 1 if group % 2 == 0 else 2
   np.testing.assert_allclose(out[group*32:(group+1)*32],expected)
 def test_q6_k_zero_block(self):
  raw=bytes(128+64+16)+np.float16(1).tobytes()
  np.testing.assert_allclose(dequantize(raw,14,(256,)),0)
 def test_q6_k_packed_quads(self):
  ql=bytes([0x98])*128; qh=bytes(64); scales=np.ones(16,dtype=np.int8).tobytes()
  raw=ql+qh+scales+np.float16(1).tobytes(); out=dequantize(raw,14,(256,))
  np.testing.assert_allclose(out[:64],-24); np.testing.assert_allclose(out[64:128],-23)
  np.testing.assert_allclose(out[128:192],-24); np.testing.assert_allclose(out[192:],-23)
 def test_direct_q4_k_matvec_matches_dequantized_reference(self):
  from engine.numba_kernels import matvec_q4_k_direct
  rng=np.random.default_rng(41); rows,cols=3,512; blocks=cols//256
  raw=bytearray(rng.integers(0,256,size=rows*blocks*144,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*144; raw[off:off+2]=np.float16(.03125).tobytes(); raw[off+2:off+4]=np.float16(.015625).tobytes()
  raw=bytes(raw); packed=np.frombuffer(raw,dtype=np.uint8).reshape(rows,blocks,144)
  params=np.frombuffer(raw,dtype='<f2').reshape(rows,blocks,72); x=rng.normal(size=cols).astype(np.float32)
  got=matvec_q4_k_direct(packed,params[:,:,0].astype(np.float32),params[:,:,1].astype(np.float32),x)
  np.testing.assert_allclose(got,dequantize(raw,12,(rows,cols))@x,rtol=2e-4,atol=2e-3)

 def test_direct_q6_k_matvec_matches_dequantized_reference(self):
  from engine.numba_kernels import matvec_q6_k_direct
  rng=np.random.default_rng(42); rows,cols=3,512; blocks=cols//256
  raw=bytearray(rng.integers(0,256,size=rows*blocks*210,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*210; raw[off+208:off+210]=np.float16(.03125).tobytes()
  raw=bytes(raw); packed=np.frombuffer(raw,dtype=np.uint8).reshape(rows,blocks,210)
  params=np.frombuffer(raw,dtype='<f2').reshape(rows,blocks,105); x=rng.normal(size=cols).astype(np.float32)
  got=matvec_q6_k_direct(packed,params[:,:,104].astype(np.float32),x)
  np.testing.assert_allclose(got,dequantize(raw,14,(rows,cols))@x,rtol=2e-4,atol=2e-3)
 def test_native_q8_0_matvec_matches_reference_when_available(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(50); rows,cols=300,512; blocks=cols//32
  raw=bytearray(rng.integers(0,256,size=rows*blocks*34,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*34; raw[off:off+2]=np.float16(.03125).tobytes()
  raw=bytes(raw); x=rng.normal(size=cols).astype(np.float32)
  ref=dequantize(raw,8,(rows,cols))@x
  for threads in (4,8):
   got=matvec(raw,x,rows,cols,8,threads)
   np.testing.assert_allclose(got,ref,rtol=2e-5,atol=2e-4)

 def test_native_q4_0_matvec_matches_reference_when_available(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(55); rows,cols=300,512; blocks=cols//32
  raw=bytearray(rng.integers(0,256,size=rows*blocks*18,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*18; raw[off:off+2]=np.float16(.03125).tobytes()
  raw=bytes(raw); x=rng.normal(size=cols).astype(np.float32)
  ref=dequantize(raw,2,(rows,cols))@x
  got=matvec(raw,x,rows,cols,2,4)
  np.testing.assert_allclose(got,ref,rtol=2e-4,atol=2e-3)

 def test_native_q5_k_matvec_matches_reference_when_available(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(54); rows,cols=300,512; blocks=cols//256
  raw=bytearray(rng.integers(0,256,size=rows*blocks*176,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    # Keep random scale/min tables but use realistic GGUF scale magnitudes.
    off=(r*blocks+b)*176; raw[off:off+2]=np.float16(.0005).tobytes(); raw[off+2:off+4]=np.float16(.00025).tobytes()
  raw=bytes(raw); x=rng.normal(size=cols).astype(np.float32)
  ref=_q8_activation_reference(raw,13,rows,cols,x)
  got=matvec(raw,x,rows,cols,13,4)
  np.testing.assert_allclose(got,ref,rtol=3e-4,atol=3e-3)

 def test_native_batched_quantized_matmul_matches_reference_when_available(self):
  from engine.native_quant import get_native, matmul
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(53); rows,cols,tokens=300,512,5
  vectors=rng.normal(size=(tokens,cols)).astype(np.float32)
  for typ,bytes_per_block,block_size,scale_offset in ((2,18,32,0),(8,34,32,0),(12,144,256,0),(13,176,256,0),(14,210,256,208)):
   blocks=cols//block_size
   raw=bytearray(rng.integers(0,256,size=rows*blocks*bytes_per_block,dtype=np.uint8).tobytes())
   for r in range(rows):
    for b in range(blocks):
     off=(r*blocks+b)*bytes_per_block+scale_offset
     raw[off:off+2]=np.float16(.03125).tobytes()
     if typ in (12,13): raw[off+2:off+4]=np.float16(.015625).tobytes()
   raw=bytes(raw)
   ref=vectors @ dequantize(raw,typ,(rows,cols)).T
   got=matmul(raw,vectors,rows,cols,typ,4)
   np.testing.assert_allclose(got,ref,rtol=3e-4,atol=3e-3,
                              err_msg=f'native batched matmul mismatch for GGML type {typ}')

 def test_native_q4_0_matvec_varied_sizes_and_nibble_edges(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(551)
  for cols in (32,64,96,512):
   rows=300; blocks=cols//32
   raw=bytearray(rng.integers(0,256,size=rows*blocks*18,dtype=np.uint8).tobytes())
   # Exercise signed nibble extrema and several finite/zero fp16 scales.
   for r in range(rows):
    for b in range(blocks):
     off=(r*blocks+b)*18
     scale=(0.0,0.0001,0.03125,1.0)[(r+b)%4]
     raw[off:off+2]=np.float16(scale).tobytes()
     if r % 3 == 0:
      raw[off+2:off+18]=bytes([0x07,0x80,0xF8,0xFF])*4
   raw=bytes(raw)
   x=rng.normal(size=cols).astype(np.float32)
   ref=dequantize(raw,2,(rows,cols))@x
   for threads in (1,4,8):
    got=matvec(raw,x,rows,cols,2,threads)
    np.testing.assert_allclose(got,ref,rtol=2e-4,atol=2e-3,
                               err_msg=f'Q4_0 matvec mismatch cols={cols}, threads={threads}')

 def test_native_q4_k_matvec_matches_reference_when_available(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  # More than 256 rows exercises the persistent native worker pool as well as
  # the AVX2 Q4_K arithmetic, including repeated calls with different worker counts.
  rng=np.random.default_rng(51); rows,cols=300,512; blocks=cols//256
  raw=bytearray(rng.integers(0,256,size=rows*blocks*144,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*144; raw[off:off+2]=np.float16(.0005).tobytes(); raw[off+2:off+4]=np.float16(.00025).tobytes()
  raw=bytes(raw); x=rng.normal(size=cols).astype(np.float32)
  ref=_q8_activation_reference(raw,12,rows,cols,x)
  for threads in (4,8):
   got=matvec(raw,x,rows,cols,12,threads)
   np.testing.assert_allclose(got,ref,rtol=2e-4,atol=2e-3)

 def test_native_q6_k_matvec_matches_reference_when_available(self):
  from engine.native_quant import get_native, matvec
  if get_native() is None: self.skipTest('native AVX2/MSVC kernel unavailable')
  rng=np.random.default_rng(52); rows,cols=300,512; blocks=cols//256
  raw=bytearray(rng.integers(0,256,size=rows*blocks*210,dtype=np.uint8).tobytes())
  for r in range(rows):
   for b in range(blocks):
    off=(r*blocks+b)*210; raw[off+208:off+210]=np.float16(.0005).tobytes()
  raw=bytes(raw); x=rng.normal(size=cols).astype(np.float32)
  ref=dequantize(raw,14,(rows,cols))@x
  for threads in (4,8):
   got=matvec(raw,x,rows,cols,14,threads)
   np.testing.assert_allclose(got,ref,rtol=2e-4,atol=2e-3)
if __name__=='__main__':unittest.main()


