<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$a = new NeuralNetwork(2, 2, 2);
$a->forwardPass([.05, .10])[0]->getValue();
//var_dump($a->forwardPass([.05, .10])[1]->getValue());

//$train = [
//    [-1, -1],
//    [1, 1],
//    [-1, -1],
//    [1, 1]
//];

//$train = [
//    [0, 0],
//    [0, 1],
//    [1, 0],
//    [1, 1]
//];

$train = [
    [.05, .10]
];

//$a->backPropagate(1, 0.5, 0.001, $train, [[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
$a->backPropagate(4, 0.5, 0.001, $train, [[.01, .99]]);
//var_dump($a->forwardPass([.05, .10]));
//var_dump($a->forwardPass([0, 1])[0]->getValue());
//var_dump($a->forwardPass([1, 0])[0]->getValue());
//var_dump($a->forwardPass([1, 1])[0]->getValue());
