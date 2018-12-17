<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$oNeuralNetwork = new NeuralNetwork(2, 20, 1);

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

$oNeuralNetwork->forwardPass($aTrainingData[0]);
$oNeuralNetwork->backPropagate(1000, 0.5, 1E-18, $aTrainingData, $aOutputData);

echo '<pre>';
print_r($oNeuralNetwork->forwardPass([1, 1]));
echo '</pre>';
