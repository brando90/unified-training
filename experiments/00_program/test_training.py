"""Deterministic correctness tests; artificial fixtures are never experiment data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
from objectives import Work,batch_tokens,causal_loss,preference_loss,loo_advantages,math_reward,extract_answer,policy_loss,prefix_reward,generate,token_logps
from training import Engine,ROUTES,Stream,seed_model
from common import METHODS,alive,identity,atomic_json
from evaluation import paired_items,score_arc_batch

torch.set_num_threads(2)

class Tokenizer:
    eos_token_id=0
    def batch_decode(self,rows,skip_special_tokens=True):return ['#### '+str(r[0]) for r in rows]

def model():
    c=GPTNeoXConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=64,rotary_pct=.25,attention_dropout=0.,hidden_dropout=0.)
    return GPTNeoXForCausalLM(c).eval()

def fixture_rows(split):
    if split.startswith('wiki'):return [{'id':str(i),'ids':[1,2,3,4,5]} for i in range(10)]
    if split.startswith('prefs'):return [{'id':str(i),'prompt_ids':[1,2],'chosen_ids':[3,4,0],'rejected_ids':[5,6,0]} for i in range(10)]
    return [{'id':str(i),'prompt_ids':[1,2],'response_ids':[3,4,0],'answer':'#### 3'} for i in range(10)]

CONFIG={'learning_rate':.001,'validation_rows':2,'batch_size':2,'rl_prompts':1,'group_size':2,'max_new_tokens':3,'budget_tokens':100000,'controller_window':4}

class ObjectiveTests(unittest.TestCase):
    def setUp(self):seed_model(0);self.m=model();self.ref=copy.deepcopy(self.m).requires_grad_(False)
    def test_prompt_padding_shift_mask(self):
        ids,attention,mask=batch_tokens([[1,2,3,4],[1,5]],[2,1],0,'cpu')
        self.assertEqual(mask.tolist(),[[False,False,True,True],[False,True,False,False]])
        self.assertEqual(attention.sum().item(),6)
        ids,attention,mask=batch_tokens([[1,2,3],[1]],[3,1],0,'cpu',left=True)
        self.assertEqual(ids.tolist(),[[1,2,3],[0,0,1]])
        self.assertFalse(mask.any())
    def test_causal_loss_matches_manual_response_shift(self):
        r=[{'prompt_ids':[1,2],'response_ids':[3,4]}];w=Work()
        loss,n=causal_loss(self.m,r,0,'cpu',w)
        logits=self.m(torch.tensor([[1,2,3,4]])).logits
        expected=torch.nn.functional.cross_entropy(logits[0,1:3],torch.tensor([3,4]))
        torch.testing.assert_close(loss,expected);self.assertEqual(w.total,4)
        loss.backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.m.parameters()))
    def test_dpo_reference_and_sign(self):
        rows=fixture_rows('prefs')[:2];w=Work();loss,n,diag=preference_loss(self.m,self.ref,rows,0,'cpu',w)
        self.assertAlmostEqual(float(loss),.69314718,places=6);self.assertEqual(diag['margin'],[0.,0.]);self.assertEqual(w.counts,{'forward':20,'reference':20})
        opt=torch.optim.SGD(self.m.parameters(),lr=.01);loss.backward();opt.step()
        loss2,_,diag=preference_loss(self.m,self.ref,rows,0,'cpu',Work())
        self.assertLess(float(loss2),float(loss));self.assertGreater(diag['margin'][0],0)
        self.assertTrue(all(p.grad is None for p in self.ref.parameters()))
    def test_unequal_length_accumulation_matches_full_batch(self):
        rows=[{'prompt_ids':[1,2],'response_ids':[3,4]}, {'prompt_ids':[1],'response_ids':[5,6,7,8]}]
        full=copy.deepcopy(self.m);micro=copy.deepcopy(self.m)
        loss,_=causal_loss(full,rows,0,'cpu',Work());loss.backward()
        for row in rows:
            part,_=causal_loss(micro,[row],0,'cpu',Work())
            (part*len(row['response_ids'])/6).backward()
        for a,b in zip(full.parameters(),micro.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
    def test_normalized_preference_cancels_uniform_language_gain_despite_length(self):
        # A uniform per-token language gain must not favor the longer answer.
        rows=[{'prompt_ids':[1],'chosen_ids':[2,3,4],'rejected_ids':[5]}]
        masks=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
        def scores(model,*args,**kwargs):
            gain=1. if model is self.m else 0.
            return torch.full((2,3),-4.+gain),masks,8
        with patch('objectives.token_logps',side_effect=scores):
            _,_,diag=preference_loss(self.m,self.ref,rows,0,'cpu',Work())
        self.assertEqual(diag['margin'],[0.])
    def test_reward_and_numeric_parser(self):
        self.assertEqual(math_reward('reason 3. Final #### 1,000.00','#### 1000'),1.)
        self.assertEqual(math_reward('no answer','#### 3'),0.)
        self.assertEqual(math_reward('#### 1/2','#### 0.5'),1.)
        self.assertIsNone(extract_answer('#### 1/0'))
    def test_rpt_leading_space_and_token_boundaries(self):
        class Fake:
            def decode(self,ids,**kwargs):return ''.join({1:' on',2:' the',3:' mat',4:'.'}[i] for i in ids)
        tok=Fake()
        self.assertEqual(prefix_reward('Prediction: on the',[1,2,3],tok),1.)
        self.assertEqual(prefix_reward('Prediction: o',[1,2,3],tok),0.)
        self.assertEqual(prefix_reward('Prediction: .',[4,1],tok),1.)
        self.assertEqual(prefix_reward('on the',[1,2,3],tok),0.)
        self.assertEqual(prefix_reward('Prediction: ',[1,2,3],tok),0.)
    def test_real_left_padded_sampling_matches_right_padded_scoring(self):
        prompts=[[1,2,3],[1]];w=Work()
        rows,_,sample_lp=generate(self.m,prompts,Tokenizer(),'cpu',w,3,True,capture_logps=True)
        lp,mask,_=token_logps(self.m,[p+y for p,y in zip(prompts,rows)],list(map(len,prompts)),0,'cpu',w)
        for i,s in enumerate(sample_lp):torch.testing.assert_close(lp[i][mask[i]],torch.tensor(s),atol=1e-5,rtol=1e-5)
    def test_batched_choice_scoring_matches_separate_questions(self):
        class Tok(Tokenizer):
            def encode(self,text,**kwargs):return [int(s) for s in text.split()]
        rows=[{'id':'a','prompt_ids':[1,2,3],'choices':{'text':['4','5 6'],'label':['A','B']},'answerKey':'B'},{'id':'b','prompt_ids':[1],'choices':{'text':['7 8 9','10','11 12'],'label':['A','B','C']},'answerKey':'A'}]
        combined=score_arc_batch(self.m,rows,Tok(),'cpu',Work())
        separate=[score_arc_batch(self.m,[row],Tok(),'cpu',Work())[0] for row in rows]
        for a,b in zip(combined,separate):
            self.assertEqual(a['correct_normalized'],b['correct_normalized']);self.assertEqual(a['correct_raw'],b['correct_raw'])
            torch.testing.assert_close(torch.tensor(a['normalized_scores']),torch.tensor(b['normalized_scores']),atol=1e-6,rtol=1e-6)
    def test_leave_one_out_is_not_biased_group_mean(self):
        r=torch.tensor([0.,1.,1.,1.]);a=loo_advantages(r,2)
        torch.testing.assert_close(a,torch.tensor([-1.,1.,0.,0.]))
        with self.assertRaises(ValueError):loo_advantages(r,1)
    def test_on_policy_gradient_and_generation_charge(self):
        def gen(*args,**kwargs):args[4].add('generation',6);return [[3,0],[5,0]],[False,False]
        w=Work()
        with patch('objectives.generate',side_effect=gen):loss,n,d=policy_loss(self.m,fixture_rows('gsm')[:1],Tokenizer(),'cpu',w,2,3)
        self.assertEqual(d['reward_mean'],.5);self.assertFalse(d['zero_gradient']);self.assertEqual(w.counts,{'generation':6,'forward':8})
        loss.backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.m.parameters()))
    def test_zero_variance_group_zero_policy_gradient(self):
        with patch('objectives.generate',return_value=([[5,0],[6,0]],[False,False])):
            loss,n,d=policy_loss(self.m,fixture_rows('gsm')[:1],Tokenizer(),'cpu',Work(),2,3)
        self.assertTrue(d['zero_gradient']);self.assertEqual(d['zero_variance_groups'],1);self.assertEqual(float(loss),0.)

class EngineTests(unittest.TestCase):
    def engine(self,method='fixed_joint'):
        seed_model(4);m=model()
        with patch('training.load_data',side_effect=fixture_rows):return Engine('01_fixture',method,4,CONFIG,'cpu',(m,copy.deepcopy(m).requires_grad_(False),Tokenizer()))
    def test_every_method_routes(self):
        self.assertTrue(set(METHODS)<=set(ROUTES)|{'parallel_joint','aioli_objective','chord_objective','rpt_inspired'})
        for method in METHODS:
            e=self.engine(method);self.assertIn(e.choose(.4),{'pt','sft','dpo','rl','rpt','hybrid','parallel'})
    def test_stream_restart_and_independent_rng(self):
        a=Stream(list(range(7)),2);a.take(5);s=a.state_dict();expected=a.take(12)
        b=Stream(list(range(7)),99);b.load_state_dict(s);self.assertEqual(b.take(12),expected)
    def test_checkpoint_exact_optimization_resume(self):
        a=self.engine();a.initialize();a.update('pt',.1)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.pt';a.checkpoint(p,{'id':'unit'})
            a.update('sft',.2);expected={k:v.clone() for k,v in a.model.state_dict().items()};cost=a.work.total;stream=a.streams['sft'].state_dict()
            b=self.engine();b.resume(p,{'id':'unit'});b.update('sft',.2)
            for k,v in b.model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertEqual(b.work.total,cost);self.assertEqual(b.streams['sft'].state_dict(),stream)
            with self.assertRaises(ValueError):b.resume(p,{'id':'wrong'})
    def test_probe_state_survives_checkpoint(self):
        a=self.engine('validation_progress');a.initialize();a.begin_probe();a.probe_state['offset']=1
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.pt';a.checkpoint(p,{'id':'probe'});b=self.engine('validation_progress');b.resume(p,{'id':'probe'})
            self.assertEqual(a.probe_state,b.probe_state);self.assertEqual(a.scheduler.snapshot(),b.scheduler.snapshot())
    def test_resume_real_adaptive_advance_including_rl(self):
        for method in ('aioli_objective','validation_progress'):
            a=self.engine(method)
            for _ in range(7):a.advance()
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'state.pt';a.checkpoint(p,{'id':method})
                expected=[a.advance() for _ in range(45)];cost=a.work.total
                weights={k:v.clone() for k,v in a.model.state_dict().items()}
                b=self.engine(method);b.resume(p,{'id':method});actual=[b.advance() for _ in range(45)]
                self.assertEqual(actual,expected);self.assertEqual(b.work.total,cost)
                for k,v in b.model.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
    def test_aioli_probe_mixtures_exact(self):
        e=self.engine('aioli_objective');e.begin_probe()
        for j,actions in zip(e.probe_state['order'],e.probe_state['actions']):
            from controller import OBJECTIVES
            self.assertEqual(actions.count(OBJECTIVES[j]),9);self.assertEqual(len(actions),12)
    def test_cost_includes_validation_and_backward(self):
        e=self.engine();e.initialize();self.assertGreater(e.work.counts['validation'],0);self.assertGreater(e.work.counts['validation_generation'],0)
        before=e.work.total;e.update('pt',.1);self.assertEqual(e.work.total-before,30);self.assertEqual(e.work.counts['backward'],20)
    def test_identity_rejects_pid_reuse(self):
        i=identity();self.assertTrue(alive(i));i['start_ticks']='0';self.assertFalse(alive(i))
    def test_paired_missing_is_not_zero(self):
        r=paired_items([{'id':'x','correct':1}],[],'correct');self.assertIn('mismatched',r['status'])

    def test_parallel_is_average_of_independent_updates_at_same_snapshot(self):
        a=self.engine('parallel_joint');b=self.engine('parallel_joint');c=self.engine('parallel_joint')
        # Fixed opposite rewards exercise nonzero policy gradients without any
        # extra stochastic fixture generation or synthetic experiment data.
        with patch('objectives.generate',return_value=([[3,0],[5,0]],[False,False])):
            b.update('sft',.4);c.update('rl',.4);a.update('parallel',.4)
        for actual,sft,rl in zip(a.model.parameters(),b.model.parameters(),c.model.parameters()):torch.testing.assert_close(actual,.5*(sft+rl),atol=1e-7,rtol=1e-6)
        self.assertTrue(a.optimizers['sft'].state);self.assertTrue(a.optimizers['rl'].state)

    def test_zero_reward_preserves_parameters_and_optimizer_state(self):
        e=self.engine();before={k:p.clone() for k,p in e.model.state_dict().items()}
        with patch('objectives.generate',return_value=([[5,0],[6,0]],[False,False])):e.update('rl',.4)
        for k,p in e.model.state_dict().items():torch.testing.assert_close(p,before[k],rtol=0,atol=0)
        self.assertFalse(e.optimizers['rl'].state)

    def test_reference_clock_matches_staged_dpo_entry(self):
        e=self.engine('sequential');e.update('sft',.74);e.refresh_reference(.75)
        for a,b in zip(e.model.parameters(),e.reference.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertIn(.75,e.reference_refreshes)
    def test_reinforcement_validation_is_sampled_success_not_gold_nll(self):
        e=self.engine()
        with patch('training.generate',return_value=([[3,0],[5,0]],[False,False])):mixed=e.validate()['rl']
        with patch('training.generate',return_value=([[5,0],[5,0]],[False,False])):zero=e.validate()['rl']
        self.assertEqual(mixed,.5);self.assertAlmostEqual(zero,2.5/3)

if __name__=='__main__':unittest.main()
