#include <iostream>
#include "NeuronioReLu.hpp"

int main() {
    
    vector<double> pesos1 = {0.2, 0.4};
    double bias1 = -1.5;
    NeuronioReLU neuronio1(pesos1, bias1);
    vector<double> entradas1 = {0.3, 2.0};
    cout << "Caso 1 - Saída: " << neuronio1.predict(entradas1) << endl;

    
    vector<double> pesos2 = {0.2, 0.4};
    double bias2 = -0.5;
    NeuronioReLU neuronio2(pesos2, bias2);
    vector<double> entradas2 = {0.3, 2.0};
    cout << "Caso 2 - Saída: " << neuronio2.predict(entradas2) << endl;

    
    vector<double> pesos3 = {0.5, 0.1};
    double bias3 = -0.2;
    NeuronioReLU neuronio3(pesos3, bias3);
    vector<double> entradas3 = {1.0, 0.5};
    cout << "Caso 3 - Saída: " << neuronio3.predict(entradas3) << endl;

    
    vector<double> pesos4 = {0.3, 0.7};
    double bias4 = -1.0;
    NeuronioReLU neuronio4(pesos4, bias4);
    vector<double> entradas4 = {0.5, 1.5};
    cout << "Caso 4 - Saída: " << neuronio4.predict(entradas4) << endl;

    return 0;
}