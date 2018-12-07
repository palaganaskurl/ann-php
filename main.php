<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$a = new NeuralNetwork(2, 1, 2, 1);
//var_dump($a->forwardPass([.05, .10]));

$train = [
    [0.01, 0.02, 0.03, 0.04],
    [0.05, 0.06, 0.07, 0.08]
];

$a->backPropagate(2, 0.5, 0.000001, $train, [1, 2]);
