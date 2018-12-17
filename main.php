<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$a = new NeuralNetwork(2, 2, 2);

$a->forwardPass([.05, .10])[0]->getValue();
echo '<pre>';
    print_r($a);
echo '</pre>';

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
$b = $a->backPropagate(5, 0.5, 0.0000001, $train, [[.01, .99]]);

echo '<pre>';
print_r($b);
print_r($a);
//print_r($a->forwardPass([.05, .10]));
echo '</pre>';
//var_dump($a->forwardPass([0, 1])[0]->getValue());
//var_dump($a->forwardPass([1, 0])[0]->getValue());
//var_dump($a->forwardPass([1, 1])[0]->getValue());
