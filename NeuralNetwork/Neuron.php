<?php

namespace NeuralNetwork;

class Neuron
{
    private $aWeights;

    private $fBias;

    private $fSum;

    private $fValue;

    public function __construct(array $aWeights, float $fBias)
    {
        $this->aWeights = $aWeights;
        $this->fBias = $fBias;
        $this->fSum = 0.0;
    }

    public function getWeights()
    {
        return $this->aWeights;
    }

    public function setWeights(array $aWeights)
    {
        $this->aWeights = $aWeights;
    }

    public function getBias()
    {
        return $this->fBias;
    }

    public function setBias(float $fBias)
    {
        $this->fBias = $fBias;
    }

    public function getSum()
    {
        return $this->fSum;
    }

    public function setSum(float $fSum)
    {
        $this->fSum += $fSum;
    }

    public function getValue()
    {
        return $this->fValue;
    }

    public function setValue(float $fValue)
    {
        $this->fValue = $fValue;
    }
}
