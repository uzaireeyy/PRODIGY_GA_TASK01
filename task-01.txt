
(base) C:\Users\Uzair>python
Python 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:03:56) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> import torch
>>> # encode context the generation is conditioned on
>>> model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'tokenizer' is not defined. Did you mean: 'AutoTokenizer'?
>>>
>>> # generate 40 new tokens
>>> greedy_output = model.generate(**model_inputs, max_new_tokens=40)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'model' is not defined
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'tokenizer' is not defined. Did you mean: 'AutoTokenizer'?
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> import torch
>>>
>>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"
>>>
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
C:\Users\Uzair\anaconda3\New folder\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
>>>
>>> # add the EOS token as PAD token to avoid warnings
>>> model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
>>> # encode context the generation is conditioned on
>>> model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)
>>>
>>> # generate 40 new tokens
>>> greedy_output = model.generate(**model_inputs, max_new_tokens=40)
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll ever be able to walk with my dog.

I'm not sure
>>>
>>> # activate beam search and early_stopping
>>> beam_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     num_beams=5,
...     early_stopping=True
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I'm not sure if I'll ever be able to walk with him again. I'm not sure
>>> # set no_repeat_ngram_size to 2
>>> beam_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     num_beams=5,
...     no_repeat_ngram_size=2,
...     early_stopping=True
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I've been thinking about this for a while now, and I think it's time for me to
>>> # set return_num_sequences > 1
>>> beam_outputs = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     num_beams=5,
...     no_repeat_ngram_size=2,
...     num_return_sequences=5,
...     early_stopping=True
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> # now we have 3 output sequences
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> for i, beam_output in enumerate(beam_outputs):
...   print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
...
0: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I've been thinking about this for a while now, and I think it's time for me to
1: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with her again.

I've been thinking about this for a while now, and I think it's time for me to
2: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I've been thinking about this for a while now, and I think it's a good idea to
3: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I've been thinking about this for a while now, and I think it's time to take a
4: I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with him again.

I've been thinking about this for a while now, and I think it's a good idea.
>>> # set seed to reproduce results. Feel free to change the seed though to get different results
>>> from transformers import set_seed
>>> set_seed(42)
>>>
>>> # activate sampling and deactivate top_k by setting top_k sampling to 0
>>> sample_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     do_sample=True,
...     top_k=0
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog but what I love about being a dog cat person is being a pet being with people who can treat you. I feel happy to be such a pet person and get to meet so many people. I
>>> # set seed to reproduce results. Feel free to change the seed though to get different results
>>> set_seed(42)
>>>
>>> # use temperature to decrease the sensitivity to low probability candidates
>>> sample_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     do_sample=True,
...     top_k=0,
...     temperature=0.6,
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog but I also love the fact that my cat is not a dog. She is a good, loving dog. I do not like to be held back by other dogs but I think that I have to
>>> # set seed to reproduce results. Feel free to change the seed though to get different results
>>> set_seed(42)
>>>
>>> # set top_k to 50
>>> sample_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     do_sample=True,
...     top_k=50
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog but what I love about being a dog is I see a beautiful pet being cared for – I love having the opportunity to see her every day so I feel very privileged to have been able to help this
>>> # set seed to reproduce results. Feel free to change the seed though to get different results
>>> set_seed(42)
>>>
>>> # set top_k to 50
>>> sample_output = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     do_sample=True,
...     top_p=0.92,
...     top_k=0
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
I enjoy walking with my cute dog but what I love about being a dog cat person is being a pet being with people who can treat you. I feel happy to be such a pet person and get to meet so many people. I
>>> # set seed to reproduce results. Feel free to change the seed though to get different results
>>> set_seed(42)
>>>
>>> # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
>>> sample_outputs = model.generate(
...     **model_inputs,
...     max_new_tokens=40,
...     do_sample=True,
...     top_k=50,
...     top_p=0.95,
...     num_return_sequences=3,
... )
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
>>>
>>> print("Output:\n" + 100 * '-')
Output:
----------------------------------------------------------------------------------------------------
>>> for i, sample_output in enumerate(sample_outputs):
...   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
...
0: I enjoy walking with my cute dog but sometimes I get nervous when she is around. I've been told that with her alone, she will usually wander off and then try to chase me. It's nice to know that I have this
1: I enjoy walking with my cute dog. I think she is the same one I like to walk with my dog, I think she is about as girly as my first dog. I hope we can find an apartment for her when we
2: I enjoy walking with my cute dog, but there's so much to say about him that I am going to miss it all. He has been so supportive and even had my number in his bag.

I hope I can say
>>>