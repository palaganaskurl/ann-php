<?php

namespace NeuralNetwork;

class NeuralNetwork
{
    /** @var Layer InputNeuron[] */
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

        for ($iOutputNeuronCount = 0; $iOutputNeuronCount < $iNumOfOutputNeurons; $iOutputNeuronCount++) {
            $aOutputNeurons[$iOutputNeuronCount] = new HiddenNeuron($this->generateRandomWeights($iNumOfHiddenNeurons), $this->randomDecimal());
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
            $oHiddenNeuron->setWeights([[.15, .20], [.25, .30]][$iIndex]);
            $oHiddenNeuron->setBias(.35);
        }

        $aOutputLayer = $this->aOutputLayer;

        /* @var $aOutputLayer OutputNeuron[] */
        foreach ($aOutputLayer as $iIndex => $oOutputNeuron) {
            $oOutputNeuron->setSumToZero();
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
        /** @var $aHiddenLayer HiddenNeuron[] */
        $aHiddenLayer = $this->aHiddenLayer;

        foreach ($aHiddenLayer as $oHiddenNeuron) {
            $aWeights = $oHiddenNeuron->getWeights();
            //var_dump('HIDDEN SA FORWARD');
            //var_dump($aWeights);

            foreach ($aWeights as $iIndex => $fWeight) {
                /** @var $oInputNeuron InputNeuron */
                $oInputNeuron = $this->aInputLayer[$iIndex];
                $oHiddenNeuron->setSum($oInputNeuron->getInput() * $fWeight);
            }

            $oHiddenNeuron->setSum($oHiddenNeuron->getBias());
            $oHiddenNeuron->activateNeuron();
            $oHiddenNeuron->setSumToZero();
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
            //var_dump('OUTPUT SA FORWARD');
            //var_dump($aWeights);

            foreach ($aWeights as $iIndex => $fWeight) {
                //var_dump('From compute: ' . $iIndex . ': ' . $fWeight);
                /** @var $oHiddenNeuron HiddenNeuron */
                $oHiddenNeuron = $this->aHiddenLayer[$iIndex];
                $oOutputNeuron->setSum($fWeight * $oHiddenNeuron->getValue());
            }

            $oOutputNeuron->setSum($oOutputNeuron->getBias());
            $oOutputNeuron->activateNeuron();
            $oOutputNeuron->setSumToZero();
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
            //$aNewHiddenNeuronWeights[] = array_fill(0, count($aInputLayer), 0.5);
            $aNewHiddenNeuronWeights[] = $oHiddenNeuron->getWeights();
        }

        foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
            //$aNewOutputNeuronWeights[] = array_fill(0, count($aHiddenLayer), 0.5); // 0 kase isa lang 'yung hidden layer
            $aNewOutputNeuronWeights[] = $oOutputNeuron->getWeights();
        }

        //$aNewOutputNeuronWeights = array_fill(0, count($this->aHiddenLayer), 0.5); // make this 2 d array
        $aNewOutputNeuronBias = array_fill(0, count($this->aOutputLayer), 0.35);
        $aNewHiddenNeuronBias = array_fill(0, count($this->aHiddenLayer), 0.60);

        $fError = 1.0;
        $iTrainingIndex = 0;

        $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
        $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;

        for ($iEpochCounter = 0; $iEpochCounter < $iEpoch; $iEpochCounter++) {
            if ($fError <= $fTargetError) break;

            //var_dump('BACK PROP HIDDEN');
            //var_dump($aOldHiddenNeuronWeights);
            //var_dump('BACK PROP OUTPUT');
            //var_dump($aOldOutputNeuronWeights);

            $aBackPropagationInputs = $aInputData[$iTrainingIndex];
            $mBackPropagationOutput = $aOutputData[$iTrainingIndex];

            foreach ($aInputLayer as $iIndex => $oInputNeuron) {
                $oInputNeuron->setInput($aBackPropagationInputs[$iIndex]);
            }

            $fError = 0.0;
            $aTotalToOut = [];
            $aOutToNet = [];
            $fMSE = 0.0;

            foreach ($aOutputLayer as $iOutputNeuronIndex => $oOutputNeuron) {
                $fOutput = $oOutputNeuron->getValue();
                //var_dump('TARGET: ' . $mBackPropagationOutput[$iOutputNeuronIndex]);
                //var_dump('OUTPUT: ' . $fOutput);
                $fMSE = .5 * pow($mBackPropagationOutput[$iOutputNeuronIndex] - $fOutput, 2);
                //var_dump('MSE: ' . $fMSE);
                // eTotal mse + mse
                $fError += $fMSE;

                // Etotal / eOut
                $aTotalToOut[] = ($mBackPropagationOutput[$iOutputNeuronIndex] - $fOutput) * -1;

                $aOutToNet[] = $fOutput * (1 - $fOutput);
            }
            var_dump('ERROR: ' . $fError);

            foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                    $fNetToWeight = $oHiddenNeuron->getValue();
                    $fTotalToWeight = $aTotalToOut[$iOutputNeuronCounter] * $aOutToNet[$iOutputNeuronCounter] * $fNetToWeight;
                    //$fLastWeight = $oOutputNeuron->getWeightAtIndex($iHiddenNeuronCounter);
                    $aNewOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter] = $aOldOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter] - ($fLearningRate * $fTotalToWeight);
                }
            }

            $aETotalToOutH = [];

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $fTotal = 0.0;

                foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                    $fOutputToHidden = $aTotalToOut[$iOutputNeuronCounter] * $aOutToNet[$iOutputNeuronCounter] * $aOldOutputNeuronWeights[$iOutputNeuronCounter][$iHiddenNeuronCounter];
                    $fTotal += $fOutputToHidden;
                }

                $aETotalToOutH[] = $fTotal;
            }

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $fOutHToNetH = $oHiddenNeuron->getValue() * (1 - $oHiddenNeuron->getValue());

                foreach ($aInputLayer as $iInputNeuronCounter => $oInputNeuron) {
                    $fETotalToWeight = $aETotalToOutH[$iHiddenNeuronCounter] * $fOutHToNetH * $oInputNeuron->getInput();
                    $aNewHiddenNeuronWeights[$iHiddenNeuronCounter][$iInputNeuronCounter] = $aOldHiddenNeuronWeights[$iHiddenNeuronCounter][$iInputNeuronCounter] - ($fLearningRate * $fETotalToWeight);
                    //$fLastWeight = $oHiddenNeuron->getWeightAtIndex($iInputNeuronCounter);
                    //$oHiddenNeuron->setWeightAtIndex($iInputNeuronCounter, $fLastWeight - ($fLearningRate * $fETotalToWeight));
                }
            }

            foreach ($aOutputLayer as $iOutputNeuronCounter => $oOutputNeuron) {
                $oOutputNeuron->setWeights($aNewOutputNeuronWeights[$iOutputNeuronCounter]);
            }

            foreach ($aHiddenLayer as $iHiddenNeuronCounter => $oHiddenNeuron) {
                $oHiddenNeuron->setWeights($aNewHiddenNeuronWeights[$iHiddenNeuronCounter]);
            }

            $this->forwardPassHelper();

            $aOldHiddenNeuronWeights = $aNewHiddenNeuronWeights;
            $aOldOutputNeuronWeights = $aNewOutputNeuronWeights;

            $iTrainingIndex++;

            if ($iTrainingIndex > count($aInputData) - 1) $iTrainingIndex = 0;
        }
    }
}
