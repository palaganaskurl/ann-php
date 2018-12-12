<?php

namespace NeuralNetwork;

class NeuralNetwork
{
    private $aInputLayer;

    private $aHiddenLayer;

    private $aOutputLayer;

    public function __construct(int $iNumOfInputs, int $iNumOfHiddenNeurons, int $iNumOfOutputNeurons)
    {
        $this->aInputLayer = new Layer($this->generateInputNeurons($iNumOfInputs));
        $this->aHiddenLayer = new Layer($this->generateHiddenNeurons($iNumOfInputs, $iNumOfHiddenNeurons));
        $this->aOutputLayer = new Layer($this->generateOutputNeurons($iNumOfHiddenNeurons, $iNumOfOutputNeurons));
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

    private function generateHiddenNeurons(int $iNumOfInputs, int $iNumOfHiddenNeurons)
    {
        $aHiddenNeurons = new Layer();

        for ($iHiddenNeuronCounter = 0; $iHiddenNeuronCounter < $iNumOfHiddenNeurons; $iHiddenNeuronCounter++) {
            $aHiddenNeurons[] = new HiddenNeuron($this->generateRandomWeights($iNumOfInputs), $this->randomDecimal());
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
        }

        /* @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        foreach ($aHiddenLayer as $iIndex => $oHiddenNeuron) {
            $oHiddenNeuron->setSumToZero();
        }

        $aOutputLayer = $this->aOutputLayer;

        /* @var $aOutputLayer OutputNeuron[] */
        foreach ($aOutputLayer as $iIndex => $oOutputNeuron) {
            $oOutputNeuron->setSumToZero();
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
        /** @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        foreach ($aHiddenLayer as $oHiddenNeuron) {
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

        foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
            $aNewHiddenNeuronWeights[] = array_fill(0, count($aInputLayer), 0.5);
        }

        $aNewOutputNeuronWeights = array_fill(0, count($this->aHiddenLayer), 0.5); // make this 2 d array
        $aNewOutputNeuronBias = array_fill(0, count($this->aOutputLayer), 0.5);
        $aNewHiddenNeuronBias = array_fill(0, count($this->aHiddenLayer), 0.5);

        $fError = 1.0;
        $iTrainingIndex = 0;

        for ($iEpochCounter = 0; $iEpochCounter < $iEpoch; $iEpochCounter++) {
            if ($fError <= $fTargetError) break;

            $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
            $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;
            $aOldHiddenNeuronBias = $aNewHiddenNeuronBias;
            $aOldOutputNeuronBias = $aNewOutputNeuronBias;

            $aOutputNeuron = $this->aOutputLayer;

            // set weights and bias to 0.5
            /** @var $aOutputNeuron OutputNeuron[] */
            foreach ($aOutputNeuron as $iOutputNeuronCounter => $oOutputNeuron) {
                $oOutputNeuron->setWeights($aOldOutputNeuronWeights);
                $oOutputNeuron->setBias($aOldOutputNeuronBias[$iOutputNeuronCounter]);
                $oOutputNeuron->setSumToZero();
            }

            // set weights and bias to 0.5
            foreach ($aHiddenLayer as $iHiddenNeuronIndex => $oHiddenNeuron) {
                $oHiddenNeuron->setWeights($aOldHiddenNeuronWeights[$iHiddenNeuronIndex]);
                $oHiddenNeuron->setBias($aOldHiddenNeuronBias[$iHiddenNeuronIndex]);
                $oHiddenNeuron->setSumToZero();
            }

            $aBackPropagationInputs = $aInputData[$iTrainingIndex];
            $mBackPropagationOutput = $aOutputData[$iTrainingIndex];

            foreach ($aInputLayer as $iIndex => $oInputNeuron) {
                $oInputNeuron->setInput($aBackPropagationInputs[$iIndex]);
            }

            $fError = 0.0;

            foreach ($aOutputLayer as $oOutputNeuron) {
                $fOutput = $oOutputNeuron->getValue();
                $fError += (1 / count($aOutputLayer)) * pow($mBackPropagationOutput - $fOutput, 2); // total error
            }

            /** @var $oOutput OutputNeuron */
            $fOutput = $this->forwardPassHelper()[0]->getValue(); // for now get the single output

            $fError = pow($mBackPropagationOutput - $fOutput, 2);

            $dE_dOut = $fOutput - $mBackPropagationOutput;
            $dOut_dNetO = 1 - pow($fOutput, 2); // Output - pow(shit) dapat nasa loob 'to ng loop per output neuron

            // dito dapat may loop ng output neuron
            /** Hidden Neuron to Output Weights */
            foreach ($aHiddenLayer as $iNeuronCounter => $oHiddenNeuron) {
                // ang nangyayari dito, sa hidden layer nagloloop pero iisang output lang
                // dapat may loop sa hidden layer tapos mag
                $dNetO_dW = $oHiddenNeuron->getValue();
                $dE_dW = $dE_dOut * $dOut_dNetO * $dNetO_dW;
                $aNewOutputNeuronWeights[$iNeuronCounter] = $aOldOutputNeuronWeights[$iNeuronCounter] - ($fLearningRate * $dE_dW);
            }

            /** @var $aOutputLayer OutputNeuron[] */
            $aOutputLayer = $this->aOutputLayer;

            /** Output Neuron Bias */
            foreach ($aOutputLayer as $iIndex => $oOutputNeuron) {
                $dNetO_dB = 1.0;
                $dE_dB = $dE_dOut * $dOut_dNetO * $dNetO_dB;
                $aNewOutputNeuronBias[$iIndex] = $aOldOutputNeuronBias[$iIndex] - ($fLearningRate * $dE_dB);
            }

            $dNetO_dOutH = [];

            // hidden neuron
            foreach ($aHiddenLayer as $iIndex => $oHiddenNeuron) {
                /** @var $oOutputNeuron OutputNeuron */
                $oOutputNeuron = $this->aOutputLayer[0];
                $dNetO_dOutH[] = $oOutputNeuron->getWeights()[$iIndex];
            }

            $dOutH_dNetOH = [];

            /** Hidden Neuron Net Value */
            foreach ($aHiddenLayer as $oHiddenNeuron) {
                $dOutH_dNetOH[] = 1 - pow($oHiddenNeuron->getValue(), 2);
            }

            $dNetOH_dW = [];

            /** NET OH WEIGHT */
            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $dNetOH_dW[] = [];

                foreach ($aInputLayer as $iInputNeuronIndex => $oInputNeuron) {
                    $dNetOH_dW[$iHiddenNeuronCounter][] = $aBackPropagationInputs[$iInputNeuronIndex];
                }
            }

            $dE_dWI = [];

            // input to hidden weight
            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $dE_dWI[] = [];

                foreach ($aInputLayer as $iInputNeuronIndex => $oInputNeuron) {
                    $dE_dWI[$iHiddenNeuronCounter][] = $dE_dOut * $dOut_dNetO * $dNetO_dOutH[$iInputNeuronIndex] * $dOutH_dNetOH[$iInputNeuronIndex] * $dNetOH_dW[$iHiddenNeuronCounter][$iInputNeuronIndex];
                }
            }

            $iNewHiddenNeuronCount = count($aNewHiddenNeuronWeights);

            for ($iCounter = 0; $iCounter < $iNewHiddenNeuronCount; $iCounter++) {
                $iInnerNeuronCount = count($aNewHiddenNeuronWeights[$iCounter]);

               for($iSubCounter = 0; $iSubCounter < $iInnerNeuronCount; $iSubCounter++) {
                   $aNewHiddenNeuronWeights[$iCounter][$iSubCounter] = $aOldHiddenNeuronWeights[$iCounter][$iSubCounter] - ($fLearningRate * $dE_dWI[$iCounter][$iSubCounter]);
               }
            }

            $dNetOH_dB = array_fill(0, count($aHiddenLayer), 1.0);

            $dE_dBI = [];

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $dE_dBI[] = $dE_dOut * $dOut_dNetO * $dNetO_dOutH[$iHiddenNeuronCounter] * $dOutH_dNetOH[$iHiddenNeuronCounter] * $dNetOH_dB[$iHiddenNeuronCounter];
            }

            $iHiddenOldBiasCount = count($aOldHiddenNeuronBias);

            for ($iCounter = 0; $iCounter < $iHiddenOldBiasCount; $iCounter++) {
                $aNewHiddenNeuronBias[$iCounter] = $aOldHiddenNeuronBias[$iCounter] - ($fLearningRate * $dE_dBI[$iCounter]);
            }

            //foreach ($aHiddenLayer as $oHiddenNeuron) {
            //    $oHiddenNeuron->setWeights()
            //}

            $iTrainingIndex++;

            if ($iTrainingIndex > count($aInputData) - 1) $iTrainingIndex = 0;
        }
    }
}
