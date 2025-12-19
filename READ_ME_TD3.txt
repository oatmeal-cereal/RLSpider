--- How to run ---
Open "TD3implementation.py"
Run it using vs code (can also try preferred IDE)
Code will initially run a rendered simulation to demonstrate the untrained model
Code will run for "EPISODES_NUMBER" episodes to train, without rendering
Once training is complete, you can press enter at the terminal to see the rendered and trained model
You can keep viewing trained simulations by pressing enter, or you can enter "e" to exit


--- Third Party Libraries ---
gymnasium - used to provide the environment for the MuJoCo walker ant
PyTorch - provides neural network support for obtaining Q-values
NumPy - Used for array operations, and for generating noise with a gaussian distribution