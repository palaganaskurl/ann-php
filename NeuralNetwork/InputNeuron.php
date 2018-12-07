<?php

namespace NeuralNetwork;

class InputNeuron
{
    private $fInput;

    public function __construct(float $fInput)
    {
        $this->fInput = $fInput;
    }

    public function getInput()
    {
        return $this->fInput;
    }

    public function setInput(float $fInput)
    {
        $this->fInput = $fInput;
    }
}
