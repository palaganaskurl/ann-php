<?php

namespace NeuralNetwork;

class NeuralNetwork
{
    private $aInputLayer;

    private $aHiddenLayer;

    private $aOutputLayer;

    public function __construct(int $iNumOfInputs, int $iNumOfHiddenLayer, int $iNumOfHiddenNeurons, int $iNumOfOutputNeurons)
    {
        $this->aInputLayer = new Layer($this->generateInputNeurons($iNumOfInputs));
        $this->aHiddenLayer = new Layer($this->generateHiddenNeurons($iNumOfInputs, $iNumOfHiddenLayer, $iNumOfHiddenNeurons));
        //foreach ($this->aHiddenLayer as $a) {
        //    foreach ($a as $b) {
        //        var_dump($b->getWeights());
        //    }
        //}die();
        $this->aOutputLayer = new Layer($this->generateOutputNeurons($iNumOfHiddenNeurons, $iNumOfOutputNeurons));
    }

    private function tansigActivationFunction(float $fSum)
    {
        return 1 / (1 + exp(0 - $fSum));
        //return (exp($fSum) - exp(0 - $fSum)) / (exp($fSum) + exp(0 - $fSum));
    }

    private function randomDecimal()
    {
        return (float)rand() / (float)getrandmax();
    }

    private function generateInputNeurons(int $iNumOfInputs)
    {
        $aInputNeurons = [];

        for ($iInputNeuronCounter = 0; $iInputNeuronCounter < $iNumOfInputs; $iInputNeuronCounter++) {
            $aInputNeurons[] = new InputNeuron($this->randomDecimal());
        }

        return $aInputNeurons;
    }

    private function generateRandomWeights(int $iNeuronCount)
    {
        $aRandomWeights = [];

        for ($iNeuronCounter = 0; $iNeuronCounter < $iNeuronCount; $iNeuronCounter++) {
            $aRandomWeights[] = $this->randomDecimal();
        }

        return $aRandomWeights;
    }

    private function generateHiddenNeurons(int $iNumOfInputs, int $iNumOfHiddenLayer, int $iNumOfHiddenNeurons)
    {
        $aHiddenNeurons = [];
        //$aRandomWeights = $this->generateRandomWeights($iNumOfInputs);

        for ($iHiddenLayerCounter = 0; $iHiddenLayerCounter < $iNumOfHiddenLayer; $iHiddenLayerCounter++) {
            $aHiddenNeurons[] = new Layer();
            $aRandomWeights = $this->generateRandomWeights($iNumOfInputs); // can be move also inside the for below

            for ($iHiddenNeuronCounter = 0; $iHiddenNeuronCounter < $iNumOfHiddenNeurons; $iHiddenNeuronCounter++) {
                $aHiddenNeurons[$iHiddenLayerCounter][] = new HiddenNeuron($aRandomWeights, $this->randomDecimal());
            }
        }

        return $aHiddenNeurons;
    }

    private function generateOutputNeurons(int $iNumOfHiddenNeurons, int $iNumOfOutputNeurons)
    {
        $aOutputNeurons = [];
        $aRandomWeights = $this->generateRandomWeights($iNumOfHiddenNeurons);

        for ($iHiddenNeuronsCount = 0; $iHiddenNeuronsCount < $iNumOfOutputNeurons; $iHiddenNeuronsCount++) {
            $aOutputNeurons[] = new HiddenNeuron($aRandomWeights, $this->randomDecimal());
        }

        return $aOutputNeurons;
    }

    public function forwardPass(array $aInput)
    {
        $aInputLayer = $this->aInputLayer;

        /* @var $aInputLayer InputNeuron[] */
        foreach ($aInputLayer as $iIndex => $oInputNeuron) {
            $oInputNeuron->setInput($aInput[$iIndex]);
            var_dump('Input Neuron ' . $iIndex . ': ' . $oInputNeuron->getInput());
        }

        foreach ($this->aHiddenLayer as $iLayerIndex => $aHiddenLayer) {
            var_dump('Hidden Layer: ' . $iLayerIndex);
            /* @var $aHiddenLayer HiddenNeuron[] */
            foreach ($aHiddenLayer as $iIndex => $oHiddenNeuron) {
                $oHiddenNeuron->setSum(0.0);
                $oHiddenNeuron->setWeights([[.15, .20], [.25, .30]][$iIndex]);
                $oHiddenNeuron->setBias(.35);
                var_dump('Hidden Neuron ' . $iIndex . ' Weight : ' . implode(' ', [[.15, .20], [.25, .30]][$iIndex]));
                var_dump('Hidden Neuron ' . $iIndex . ' Bias : ' . .35);
            }
        }

        $aOutputLayer = $this->aOutputLayer;

        /* @var $aOutputLayer OutputNeuron[] */
        foreach ($aOutputLayer as $iIndex => $oOutputNeuron) {
            $oOutputNeuron->setSum(0.0);
            $oOutputNeuron->setWeights([[.40, .45], [.50, .55]][$iIndex]);
            $oOutputNeuron->setBias(.60);
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

    private function computeHiddenLayerNeuronValue()
    {
        foreach ($this->aHiddenLayer as $aHiddenLayer) {
            /** @var $aHiddenLayer HiddenNeuron[] */
            foreach ($aHiddenLayer as $oHiddenNeuron) {
                $aWeights = $oHiddenNeuron->getWeights();

                foreach ($aWeights as $iIndex => $fWeight) {
                    /** @var $oInputNeuron InputNeuron */
                    $oInputNeuron = $this->aInputLayer[$iIndex];
                    $oHiddenNeuron->setSum($oInputNeuron->getInput() * $fWeight);
                }

                $oHiddenNeuron->setSum($oHiddenNeuron->getBias());
                $oHiddenNeuron->setValue($this->tansigActivationFunction($oHiddenNeuron->getSum()));
            }
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
                $oHiddenNeuron = $this->aHiddenLayer[count($this->aHiddenLayer) - 1][$iIndex];
                $oOutputNeuron->setSum($fWeight * $oHiddenNeuron->getValue());
            }

            $oOutputNeuron->setSum($oOutputNeuron->getBias());
            $oOutputNeuron->setValue($this->tansigActivationFunction($oOutputNeuron->getSum()));
        }

        return $aOutputLayer;
    }

    public function backPropagate(int $iEpoch, float $fLearningRate, float $fTargetError, array $aInputData, array $aOutputData)
    {
        $aNewHiddenNeuronWeights = [];

        foreach ($this->aHiddenLayer as $aHiddenLayer) {
            foreach ($aHiddenLayer as $iHiddenNeuronCount => $oHiddenNeuron) {
                $aNewHiddenNeuronWeights[$iHiddenNeuronCount][] = array_fill(0, count($this->aInputLayer), 0.5);
            }
        }
        print_r($aNewHiddenNeuronWeights);

        $aNewOutputNeuronWeights = array_fill(0, count($this->aHiddenLayer), 0.5);
        $aNewOutputNeuronBias = array_fill(0, count($this->aOutputLayer), 0.5);
        $aNewHiddenNeuronBias = array_fill(0, count($this->aHiddenLayer), 0.5);

        $aBackPropagationInputs = [];
        $fError = 1.0;
        $iTrainingIndex = 0;

        for ($iEpochCounter = 0; $iEpochCounter < $iEpoch; $iEpochCounter++) {
            if ($fError <= $fTargetError) break;

            $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
            $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;
            $aOldHiddenNeuronBias = $aNewHiddenNeuronBias;
            $aOldOutputNeuronBias = $aNewOutputNeuronBias;

            $aOutputNeuron = $this->aOutputLayer;

            /** @var $aOutputNeuron OutputNeuron[] */
            foreach ($aOutputNeuron as $iOutputNeuronCounter => $oOutputNeuron) {
                $oOutputNeuron->setWeights($aOldOutputNeuronWeights);
                $oOutputNeuron->setBias($aOldOutputNeuronBias[$iOutputNeuronCounter]);
                $oOutputNeuron->setSum(0.0);
            }

            foreach ($this->aHiddenLayer as $iHiddenLayerCount => $aHiddenLayer) {
                /** @var $aHiddenLayer HiddenNeuron[] */
                foreach ($aHiddenLayer as $oHiddenNeuron) {
                    $oHiddenNeuron->setWeights($aOldHiddenNeuronWeights[$iHiddenLayerCount]);
                    $oHiddenNeuron->setBias($aOldHiddenNeuronBias[$iHiddenLayerCount]);
                    $oHiddenNeuron->setSum(0.0);
                }
            }

            $aBackPropagationInputs = $aInputData[$iTrainingIndex];

            $aInputLayer = $this->aInputLayer;

            /** @var $aInputLayer InputNeuron[]] */
            foreach ($aInputLayer as $iIndex => $oInputNeuron) {
                $oInputNeuron->setInput($aBackPropagationInputs[$iIndex]);
            }

            $mBackPropagationOutput = $aOutputData[$iTrainingIndex];

            /** @var $oOutput OutputNeuron */
            $fOutput = $this->forwardPassHelper()[0]->getValue(); // for now get the single output

            $fError = 0.5 * pow($mBackPropagationOutput - $fOutput, 2);

            $fDerivativeOfErrorWithRespectToOutput = $fOutput - $mBackPropagationOutput;
            $fDerivativeOfOutputWithRespectToNetOutput = 1 - pow($fOutput, 2);

            $iWeightCounter = 0;

            foreach ($this->aHiddenLayer as $aHiddenLayer) {
                foreach ($aHiddenLayer as $oHiddenNeuron) {
                    $fDerivativeOfNetOutputWithRespectToThisWeight = $oHiddenNeuron->getValue();
                    $fDerivativeOfErrorWithRespectToWeight = $fDerivativeOfErrorWithRespectToOutput * $fDerivativeOfOutputWithRespectToNetOutput * $fDerivativeOfNetOutputWithRespectToThisWeight;
                    $aNewOutputNeuronWeights[$iWeightCounter] = $aOldOutputNeuronWeights[$iWeightCounter] - ($fLearningRate * $fDerivativeOfErrorWithRespectToWeight);
                    $iWeightCounter++;
                }
            }

            $iWeightCounter = 0;
        }
    }
}
