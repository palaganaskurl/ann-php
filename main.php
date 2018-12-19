<?php

require __DIR__ . '/vendor/autoload.php';

use NeuralNetwork\NeuralNetwork;

$a = new NeuralNetwork(2, 2, 2, [
    'hidden' => [
        'weights' => [
            [.15, .20],
            [.25, .30]
        ],
        'bias'    => .35
    ],
    'output' => [
        'weights' => [
            [.40, .45],
            [.50, .55]
        ],
        'bias'    => .60
    ]
]);

$train = [
    [.05, .10]
];

//$a->backPropagate(1, 0.5, 0.001, $train, [[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
$b = $a->backPropagate(4, 0.5, 0.0000001, $train, [[.01, .99]]);

echo '<pre>';
print_r($b);
//print_r($a->forwardPass([.05, .10]));
echo '</pre>';
