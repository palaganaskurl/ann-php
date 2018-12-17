<?php

namespace NeuralNetwork;

/**
 * Class NeuralNetwork
 * @author Kurl Angelo D. Palaganas <kurl@simplexi.com.ph>
 * @package NeuralNetwork
 */
class NeuralNetwork
{
    /** @var Layer InputNeuron[] */
    private $aInputLayer;

    /** @var Layer HiddenNeuron[] */
    private $aHiddenLayer;

    /** @var Layer OutputLayer */
    private $aOutputLayer;

    /**
     * NeuralNetwork constructor.
     * @param int $iNumOfInputs
     * @param int $iNumOfHiddenNeurons
     * @param int $iNumOfOutputNeurons
     */
    public function __construct(int $iNumOfInputs, int $iNumOfHiddenNeurons, int $iNumOfOutputNeurons)
    {
        $this->aInputLayer = new Layer($this->generateInputNeurons($iNumOfInputs));
        $this->aHiddenLayer = new Layer($this->generateHiddenNeurons($iNumOfInputs, $iNumOfHiddenNeurons));
        $this->aOutputLayer = new Layer($this->generateOutputNeurons($iNumOfHiddenNeurons, $iNumOfOutputNeurons));
    }

    /**
     * Generates random decimal
     * @return float|int
     */
    private function randomDecimal()
    {
        return (float)rand() / (float)getrandmax();
    }

    /**
     * Generates random weights based on number of neurons of a layer.
     * @param int $iNeuronCount
     * @return array
     */
    private function generateRandomWeights(int $iNeuronCount)
    {
        $aRandomWeights = [];

        for ($iNeuronCounter = 0; $iNeuronCounter < $iNeuronCount; $iNeuronCounter++) {
            $aRandomWeights[] = $this->randomDecimal();
            //$aRandomWeights[] = .5;
        }

        return $aRandomWeights;
    }

    /**
     * Generates input neurons based on the number of input neurons.
     * @param int $iNumOfInputs
     * @return array
     */
    private function generateInputNeurons(int $iNumOfInputs)
    {
        $aInputNeurons = [];

        for ($iInputNeuronCounter = 0; $iInputNeuronCounter < $iNumOfInputs; $iInputNeuronCounter++) {
            $aInputNeurons[] = new InputNeuron($this->randomDecimal());
        }

        return $aInputNeurons;
    }

    /**
     * Generates hidden neurons based on number of hidden neurons, weights based on number of inputs.
     * @param int $iNumOfInputs
     * @param int $iNumOfHiddenNeurons
     * @return array|Layer
     */
    private function generateHiddenNeurons(int $iNumOfInputs, int $iNumOfHiddenNeurons)
    {
        $aHiddenNeurons = [];
        $fBias = 0.5;
        //$fBias = $this->randomDecimal();

        for ($iHiddenNeuronCounter = 0; $iHiddenNeuronCounter < $iNumOfHiddenNeurons; $iHiddenNeuronCounter++) {
            $aHiddenNeurons[] = new HiddenNeuron($this->generateRandomWeights($iNumOfInputs), $fBias);
            //$aHiddenNeurons[] = new HiddenNeuron($this->generateRandomWeights($iNumOfInputs), $fBias);
        }

        return $aHiddenNeurons;
    }

    /**
     * Generates output neurons based on number of output neurons, weights based on number of hidden neurons.
     * @param int $iNumOfHiddenNeurons
     * @param int $iNumOfOutputNeurons
     * @return array
     */
    private function generateOutputNeurons(int $iNumOfHiddenNeurons, int $iNumOfOutputNeurons)
    {
        $aOutputNeurons = [];
        $fBias = 0.5;
        //$fBias = $this->randomDecimal();

        for ($iOutputNeuronCount = 0; $iOutputNeuronCount < $iNumOfOutputNeurons; $iOutputNeuronCount++) {
            $aOutputNeurons[$iOutputNeuronCount] = new OutputNeuron($this->generateRandomWeights($iNumOfHiddenNeurons), $fBias);
        }

        return $aOutputNeurons;
    }

    public function forwardPass(array $aInput)
    {
        $aInputLayer = $this->aInputLayer;

        /* @var $aInputLayer InputNeuron[] */
        foreach ($aInputLayer as $iIndex => $oInputNeuron) {
            $oInputNeuron->setInput($aInput[$iIndex]);
        }

        /* @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        foreach ($aHiddenLayer as $iIndex => $oHiddenNeuron) {
            $oHiddenNeuron->setSumToZero();
            //$oHiddenNeuron->setWeights([[.15, .20], [.25, .30]][$iIndex]);
            //$oHiddenNeuron->setBias(.35);
        }

        $aOutputLayer = $this->aOutputLayer;

        /* @var $aOutputLayer OutputNeuron[] */
        foreach ($aOutputLayer as $iIndex => $oOutputNeuron) {
            $oOutputNeuron->setSumToZero();
            //$oOutputNeuron->setWeights([[.40, .45], [.50, .55]][$iIndex]);
            //$oOutputNeuron->setBias(.60);
        }

        return $this->forwardPassHelper();
    }

    /**
     * @return Layer|OutputNeuron[]
     */
    private function forwardPassHelper()
    {
        $this->computeHiddenLayerNeuronValue();

        return $this->computeOutputNeuronValue();
    }

    /**
     * Computes hidden neuron value.
     */
    private function computeHiddenLayerNeuronValue()
    {
        /** @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        foreach ($aHiddenLayer as $i => $oHiddenNeuron) {
            $aWeights = $oHiddenNeuron->getWeights();

            foreach ($aWeights as $iIndex => $fWeight) {
                /** @var $oInputNeuron InputNeuron */
                $oInputNeuron = $this->aInputLayer[$iIndex];
                $oHiddenNeuron->setSum($oInputNeuron->getInput() * $fWeight);
            }

            $oHiddenNeuron->setSum($oHiddenNeuron->getBias());
            $oHiddenNeuron->activateNeuron();
        }
    }

    /**
     * @return Layer|OutputNeuron[]
     */
    public function computeOutputNeuronValue()
    {
        $aOutputLayer = $this->aOutputLayer;

        /** @var $aOutputLayer OutputNeuron[] */
        foreach ($aOutputLayer as $oOutputNeuron) {
            $aWeights = $oOutputNeuron->getWeights();

            foreach ($aWeights as $iIndex => $fWeight) {
                /** @var $oHiddenNeuron HiddenNeuron */
                $oHiddenNeuron = $this->aHiddenLayer[$iIndex];
                $oOutputNeuron->setSum($fWeight * $oHiddenNeuron->getValue());
            }

            $oOutputNeuron->setSum($oOutputNeuron->getBias());
            $oOutputNeuron->activateNeuron();
        }

        return $aOutputLayer;
    }

    public function backPropagate(int $iEpoch, float $fLearningRate, float $fTargetError, array $aInputData, array $aOutputData)
    {
        /** @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        /** @var $aInputLayer InputNeuron[] */
        $aInputLayer = $this->aInputLayer;

        /** @var $aOutputLayer OutputNeuron[] */
        $aOutputLayer = $this->aOutputLayer;

        $aNewHiddenNeuronWeights = [];
        $aNewOutputNeuronWeights = [];

        foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
            $aNewHiddenNeuronWeights[] = $oHiddenNeuron->getWeights();
        }

        foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
            $aNewOutputNeuronWeights[] = $oOutputNeuron->getWeights();
        }

        $iTrainingIndex = 1;

        $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
        $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;

        for ($iEpochCounter = 0; $iEpochCounter < $iEpoch; $iEpochCounter++) {
            $aBackPropagationInputs = $aInputData[$iTrainingIndex];
            $mBackPropagationOutput = $aOutputData[$iTrainingIndex];

            foreach ($aInputLayer as $iIndex => $oInputNeuron) {
                $oInputNeuron->setInput($aBackPropagationInputs[$iIndex]);
            }

            $fErrorTotal = 0.0;
            $aErrorTotalToOut = [];
            $aOutToNet = [];

            foreach ($aOutputLayer as $iOutputNeuronIndex => $oOutputNeuron) {
                $fOutput = $oOutputNeuron->getValue();
                $fSquaredError = .5 * pow($mBackPropagationOutput[$iOutputNeuronIndex] - $fOutput, 2);
                $fErrorTotal += $fSquaredError;
                $aErrorTotalToOut[] = ($mBackPropagationOutput[$iOutputNeuronIndex] - $fOutput) * -1;
                $aOutToNet[] = $fOutput * (1 - $fOutput);
            }

            var_dump($iEpochCounter . ' Error: ' . $fErrorTotal);

            foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                    $fNetToWeight = $oHiddenNeuron->getValue();
                    $fTotalToWeight = $aErrorTotalToOut[$iOutputNeuronCounter] * $aOutToNet[$iOutputNeuronCounter] * $fNetToWeight;
                    $aNewOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter] = $aOldOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter] - ($fLearningRate * $fTotalToWeight);
                }
            }

            $aETotalToOutH = [];

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $fTotal = 0.0;

                foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                    $fOutputToHidden = $aErrorTotalToOut[$iOutputNeuronCounter] * $aOutToNet[$iOutputNeuronCounter] * $aOldOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter];
                    $fTotal += $fOutputToHidden;
                }

                $aETotalToOutH[] = $fTotal;
            }

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $fOutHToNetH = $oHiddenNeuron->getValue() * (1 - $oHiddenNeuron->getValue());
                //$fOutHToNetH = 1 - pow($oHiddenNeuron->getValue(), 2);

                foreach ($aInputLayer as $iInputNeuronCounter => $oInputNeuron) {
                    $fETotalToWeight = $aETotalToOutH[$iHiddenNeuronCounter] * $fOutHToNetH * $oInputNeuron->getInput();
                    $aNewHiddenNeuronWeights[$iHiddenNeuronCounter][$iInputNeuronCounter] = $aOldHiddenNeuronWeights[$iHiddenNeuronCounter][$iInputNeuronCounter] - ($fLearningRate * $fETotalToWeight);
                }
            }

            foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                //var_dump($aNewOutputNeuronWeights[$iOutputNeuronCounter]);
                $oOutputNeuron->setWeights($aNewOutputNeuronWeights[$iOutputNeuronCounter]);
                //print_r($oOutputNeuron->getWeights());
            }

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $oHiddenNeuron->setWeights($aNewHiddenNeuronWeights[$iHiddenNeuronCounter]);
            }

            if ($fErrorTotal <= $fTargetError) break;

            $this->forwardPassHelper();

            $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
            $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;

            $iTrainingIndex++;

            if ($iTrainingIndex > count($aInputData) - 1) $iTrainingIndex = 0;
        }

        return $this;
    }
}
