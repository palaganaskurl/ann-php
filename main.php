<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$a = new NeuralNetwork(2, 2, 2);
//var_dump($a->forwardPass([.05, .10]));

//$train = [
//    [-1, -1],
//    [1, 1],
//    [-1, -1],
//    [1, 1]
//];

$train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

$a->backPropagate(50000, 0.5, 0.001, $train, [1, 10, 11, 100]);
var_dump($a->forwardPass([0, 0])[0]->getValue());
var_dump($a->forwardPass([0, 1])[0]->getValue());
var_dump($a->forwardPass([1, 0])[0]->getValue());
var_dump($a->forwardPass([1, 1])[0]->getValue());
