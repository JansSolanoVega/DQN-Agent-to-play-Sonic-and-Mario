# DQN agent to play Sonic The Hedgehog and Super Mario Bros

<p align="center">
  <i>Trained models</i><br/>
  <img src="results\sonic.gif" width="200" height="200">
  <img src="results\mario.gif" width="200" height="200">
</p>

## To test games:

```sh
python -m test_games.sonic_play --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1 --scenario contest
```

```sh
python -m test_games.mario_play --game SuperMarioBros-v0 --state RIGHT_ONLY
```
## To train a new model:
*   Change hyperparameters in `params/` folder.
*   Run `./training_agent.ipynb`.

## To run trained model:
*   GAME_TO_RUN: `mario` or `sonic` 
```sh
python replay_model.py --game GAME_TO_RUN
```

