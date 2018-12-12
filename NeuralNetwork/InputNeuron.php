<?php

namespace NeuralNetwork;

/**
 * Class InputNeuron
 * @author Kurl Angelo D. Palaganas <kurl@simplexi.com.ph>
 * @package NeuralNetwork
 */
class InputNeuron
{
    /**
     * Input value
     * @var float
     */
    private $fInput;

    /**
     * InputNeuron constructor.
     * @param float $fInput
     */
    public function __construct(float $fInput)
    {
        $this->fInput = $fInput;
    }

    /**
     * Gets the input
     * @return float
     */
    public function getInput()
    {
        return $this->fInput;
    }

    /**
     * Sets input
     * @param float $fInput
     */
    public function setInput(float $fInput)
    {
        $this->fInput = $fInput;
    }
}
