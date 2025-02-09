#ifndef NEURONIO_RELU_HPP
#define NEURONIO_RELU_HPP

#include "neuronio.hpp"

class NeuronioReLU : public Neuronio {
public:
    NeuronioReLU(const vector<double>& pesos, double bias) 
        : Neuronio(pesos, bias) {}

    double predict(const vector<double> entradas) const override { 
        double soma = 0.0;
        for (size_t i = 0; i < entradas.size(); ++i) {
            soma += entradas[i] * pesos[i];
        }
        soma += bias;

        return (soma > 0.0) ? soma : 0.0;
    }
};

#endif
