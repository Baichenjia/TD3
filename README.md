## TD3 / DDPG implementation with TF-2

This implementation is inspired by OpenAI spinning up.

[Document of DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

[Document of TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)


### DDPG
The implementation of DDPG only contains *one*
script: `DDPG/main.py`.

To train a DDPG agent, write `ddpg(env_name="HalfCheetah-v2")` 
in main function of `DDPG/main.py`.

To play with pre-trained model, write 
`play(model_path="model/model_60.h5")` in the main
function of `DDPG/main.py`

### TD3
The implementation of TD3 only contains *one*
script: `TD3/main.py`.

To train a TD3 agent, write `td3(env_name="HalfCheetah-v2")` 
in main function of `TD3/main.py`.

To play with pre-trained model, write 
`play(model_path="model/model_30.h5")` in the main
function of `TD3/main.py`.