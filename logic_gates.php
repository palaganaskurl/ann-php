<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$oNeuralNetwork = new NeuralNetwork(2, 5, 1);

$aTrainingData = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];
$aOutputData = [
    [0],
    [0],
    [0],
    [1]
];

$oNeuralNetwork->backPropagate(100000, 0.5, 1E-18, $aTrainingData, $aOutputData);

print_r($oNeuralNetwork);

var_dump($oNeuralNetwork->forwardPass([0, 0])[0]->getValue());
var_dump($oNeuralNetwork->forwardPass([0, 1])[0]->getValue());
var_dump($oNeuralNetwork->forwardPass([1, 0])[0]->getValue());
var_dump($oNeuralNetwork->forwardPass([1, 1])[0]->getValue());
