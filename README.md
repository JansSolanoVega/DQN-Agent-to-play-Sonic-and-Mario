# DQN agent to play Sonic The Hedgehog and Super Mario Bros

## To test games:
```sh
python -m test_games.sonic_play --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1 --scenario contest
```

```sh
python -m test_games.mario_play --game SuperMarioBros-v0 --state RIGHT_ONLY
```
## To train a new model:
*   Change hyperparameters in `params/` folder.
*   Run `./TestingTrainingStableBaselines.ipynb`.

## To run trained model:
*   GAME_TO_RUN: `mario` or `sonic` 
```sh
python replay_model.py --game GAME_TO_RUN
```

