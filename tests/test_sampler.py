import unittest, numpy as np
from engine.sampler import sample
class SamplerTests(unittest.TestCase):
 def test_greedy(self):self.assertEqual(sample(np.array([0.,4.,2.]),temperature=0),1)
 def test_repetition_penalty(self):self.assertEqual(sample(np.array([9.,1.]),history=[0],temperature=0,repetition_penalty=2),0)
 def test_stop_token_can_be_exempt_from_prompt_repetition_penalty(self):
  logits=np.array([1., 1.5],dtype=np.float32)
  self.assertEqual(sample(logits,history=[1],temperature=0,repetition_penalty=2,penalty_exempt={1}),1)
 def test_top_k_keeps_only_highest_logit(self):self.assertEqual(sample(np.array([1.,5.,4.]),temperature=.8,top_p=1,top_k=1),1)
 def test_rejects_non_finite_logits(self):
  with self.assertRaisesRegex(ValueError, 'non-finite logits'): sample(np.array([0., np.nan]))
 def test_rejects_invalid_sampling_parameters(self):
  with self.assertRaises(ValueError): sample(np.array([1., 0.]), top_p=0)
  with self.assertRaises(ValueError): sample(np.array([1., 0.]), repetition_penalty=0)
  with self.assertRaises(ValueError): sample(np.array([1., 0.]), top_k=-1)
if __name__=='__main__':unittest.main()
