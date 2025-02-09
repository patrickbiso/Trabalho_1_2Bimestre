#include <iostream>
#include <vector>
#include "NeuronioReLu.hpp"

using namespace std;

int main() {

    vector<double> pesos1 = {0.2, 0.4};
    double bias1 = -1.5;
    Neuronio* neuronio1 = new NeuronioReLU(pesos1, bias1);
    vector<double> entradas1 = {0.3, 2.0};
    cout << "Caso 1 - Saída: " << neuronio1->predict(entradas1) << endl;
    delete neuronio1;

    vector<double> pesos2 = {0.2, 0.4};
    double bias2 = -0.5;
    Neuronio* neuronio2 = new NeuronioReLU(pesos2, bias2);
    vector<double> entradas2 = {0.3, 2.0};
    cout << "Caso 2 - Saída: " << neuronio2->predict(entradas2) << endl;
    delete neuronio2;

    vector<double> pesos3 = {0.5, 0.1};
    double bias3 = -0.2;
    Neuronio* neuronio3 = new NeuronioReLU(pesos3, bias3);
    vector<double> entradas3 = {1.0, 0.5};
    cout << "Caso 3 - Saída: " << neuronio3->predict(entradas3) << endl;
    delete neuronio3;

    vector<double> pesos4 = {0.3, 0.7};
    double bias4 = -1.0;
    Neuronio* neuronio4 = new NeuronioReLU(pesos4, bias4);
    vector<double> entradas4 = {0.5, 1.5};
    cout << "Caso 4 - Saída: " << neuronio4->predict(entradas4) << endl;
    delete neuronio4;

    
    vector<double> pesos5 = {0.1, -0.3, 0.5}; 
    double bias5 = 0.2;
    Neuronio* neuronio5 = new NeuronioReLU(pesos5, bias5);
    vector<double> entradas5 = {1.0, -1.0, 0.5};
    cout << "Caso 5 - Saída: " << neuronio5->predict(entradas5) << endl;
    delete neuronio5;

    vector<double> pesos6 = {0.0, 0.0}; //pesos nulos
    double bias6 = 0.5;
    Neuronio* neuronio6 = new NeuronioReLU(pesos6, bias6);
    vector<double> entradas6 = {2.0, -2.0};
    cout << "Caso 6 - Saída: " << neuronio6->predict(entradas6) << endl;
    delete neuronio6;

    vector<double> pesos7 = {-0.2, -0.4}; //pesos negativos
    double bias7 = 0.1;
    Neuronio* neuronio7 = new NeuronioReLU(pesos7, bias7);
    vector<double> entradas7 = {-0.5, -1.5};
    cout << "Caso 7 - Saída: " << neuronio7->predict(entradas7) << endl;
    delete neuronio7;

    return 0;
}
