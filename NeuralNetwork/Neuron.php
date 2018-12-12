<?php

namespace NeuralNetwork;

/**
 * Class Neuron
 * @author Kurl Angelo D. Palaganas <kurl@simplexi.com.ph>
 * @package NeuralNetwork
 */
class Neuron
{
    /**
     * Neuron weights
     * Represents weights connected to each neuron.
     * @var array
     */
    private $aWeights;

    /**
     * Neuron bias
     * @var float
     */
    private $fBias;

    /**
     * Neuron sum
     * @var float
     */
    private $fSum;

    /**
     * Neuron value
     * @var float
     */
    private $fValue;

    /**
     * Neuron constructor.
     * @param array $aWeights
     * @param float $fBias
     */
    public function __construct(array $aWeights, float $fBias)
    {
        $this->aWeights = $aWeights;
        $this->fBias = $fBias;
        $this->fSum = 0.0;
    }

    /**
     * Gets neuron weights
     * @return array
     */
    public function getWeights()
    {
        return $this->aWeights;
    }

    /**
     * Set neuron weights
     * @param array $aWeights
     */
    public function setWeights(array $aWeights)
    {
        $this->aWeights = $aWeights;
    }

    public function setWeightAtIndex(int $iIndex, float $fValue)
    {
        $this->aWeights[$iIndex] = $fValue;
    }

    public function getWeightAtIndex(int $iIndex)
    {
        return $this->aWeights[$iIndex];
    }

    /**
     * Gets neuron bias
     * @return float
     */
    public function getBias()
    {
        return $this->fBias;
    }

    /**
     * Sets neuron bias
     * @param float $fBias
     */
    public function setBias(float $fBias)
    {
        $this->fBias = $fBias;
    }

    /**
     * Gets the neuron sum
     * @return float
     */
    public function getSum()
    {
        return $this->fSum;
    }

    /**
     * Sets the neuron sum.
     * @param float $fSum
     */
    public function setSum(float $fSum)
    {
        $this->fSum += $fSum;
    }

    /**
     * Sets the sum to zero.
     */
    public function setSumToZero()
    {
        $this->fSum = 0.0;
    }

    /**
     * Gets neuron value.
     * @return float
     */
    public function getValue()
    {
        return $this->fValue;
    }

    /**
     * Sets neuron value.
     * @param float $fValue
     */
    private function setValue(float $fValue)
    {
        $this->fValue = $fValue;
    }

    private function activate(float $fSum)
    {
        //var_dump($fSum);
        //return $fSum;
        return 1 / (1 + exp(0 - $fSum));
        //return (exp($fSum) - exp(0 - $fSum)) / (exp($fSum) + exp(0 - $fSum));
    }

    public function activateNeuron()
    {
        $this->setValue($this->activate($this->getSum()));
    }
}
