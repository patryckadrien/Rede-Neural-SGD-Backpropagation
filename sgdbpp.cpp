//
//  Criado por Santiago Becerra em 15/9/19.
//  Copyright © 2019 Santiago Becerra. All rights reserved.
//  Link: https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
//
//  Aptado e modificado por Patryck Adrien.
//

#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>
#include <time.h>

// Rede Neural para aprfimizado das portas lógicas XOR
// Características : função de ativacao sigmoid, descida de gradiente estocástico (SGD) 
// e método de erro quadrático médio (MSE) para cálculo do erro

double sigmoid(double x) { 
    return 1. / (1. + exp(-x)); 
}

double derivada_sigmoid(double x) { 
    return x * (1. - x); 
}

double inicia_pesos() { 
    return ((double)rand())/((double)RAND_MAX);
}

void embaralhar(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(int argc, const char * argv[]) {

    double tempo_gasto = 0.;
 
    clock_t inicio = clock();

    static const int numEntradas = 2;
    static const int numNosOcultos = 2;
    static const int numSaidas = 1;
    static const int epocas = 1e5;
    
    const double lr = 0.1;
    
    double camadasOcultas[numNosOcultos];
    double camadaSaida[numSaidas];
    
    double camadasOcultasBias[numNosOcultos];
    double camadaSaidaBias[numSaidas];

    double pesosOcultos[numEntradas][numNosOcultos];
    double pesosSaida[numNosOcultos][numSaidas];
    
    static const int numSetTreinamento = 4;

    double entrada_treinamento[numSetTreinamento][numEntradas] = { 
        {0.,0.},{1.,0.},{0.,1.},{1.,1.} 
    };

    double saida_treinamento[numSetTreinamento][numSaidas] = { 
        {0.},{1.},{1.},{0.} 
    };
    
    for (int i=0; i<numEntradas; i++) {
        for (int j=0; j<numNosOcultos; j++) {
            pesosOcultos[i][j] = inicia_pesos();
        }
    }
    for (int i=0; i<numNosOcultos; i++) {
        camadasOcultasBias[i] = inicia_pesos();
        for (int j=0; j<numSaidas; j++) {
            pesosSaida[i][j] = inicia_pesos();
        }
    }
    for (int i=0; i<numSaidas; i++) {
        camadaSaidaBias[i] = inicia_pesos();
    }
    
    int OrdemSetTreinamento[] = {0,1,2,3};
    
    for (int n=0; n<epocas; n++) {
        embaralhar(OrdemSetTreinamento, numSetTreinamento);
        for (int x=0; x<numSetTreinamento; x++) {
            
            int i = OrdemSetTreinamento[x];
            
            // Propagação para frente (Forward)
            
            for (int j=0; j<numNosOcultos; j++) {
                double ativacao=camadasOcultasBias[j];
                 for (int k=0; k<numEntradas; k++) {
                    ativacao+=entrada_treinamento[i][k]*pesosOcultos[k][j]; // para otimizar transponhe a matriz pesos
                }
                camadasOcultas[j] = sigmoid(ativacao);
            }
            
            for (int j=0; j<numSaidas; j++) {
                double ativacao=camadaSaidaBias[j];
                for (int k=0; k<numNosOcultos; k++) {
                    ativacao+=camadasOcultas[k]*pesosSaida[k][j]; // para otimizar transponhe a matriz pesos
                }
                camadaSaida[j] = sigmoid(ativacao);
            }
            
            //std::cout << "Entrada:" << entrada_treinamento[i][0] << " " << entrada_treinamento[i][1] << "    Saída:" << camadaSaida[0] << "    Saída Esperada: " << saida_treinamento[i][0] << "\n";
            
            // Retropropagação (Backpropagation)
            
            double deltaSaida[numSaidas];
            for (int j=0; j<numSaidas; j++) { 
                double ErroSaida = (saida_treinamento[i][j]-camadaSaida[j]);
                deltaSaida[j] = ErroSaida*derivada_sigmoid(camadaSaida[j]);
            }
            
            double deltaOculto[numNosOcultos];
            for (int j=0; j<numNosOcultos; j++) {
                double erroOculto = 0.;
                for(int k=0; k<numSaidas; k++) {
                    erroOculto+=deltaSaida[k]*pesosSaida[j][k];
                }
                deltaOculto[j] = erroOculto*derivada_sigmoid(camadasOcultas[j]);
            }
            
            for (int j=0; j<numSaidas; j++) {
                camadaSaidaBias[j] += deltaSaida[j]*lr;
                for (int k=0; k<numNosOcultos; k++) {
                    pesosSaida[k][j]+=camadasOcultas[k]*deltaSaida[j]*lr;
                }
            }
            
            for (int j=0; j<numNosOcultos; j++) {
                camadasOcultasBias[j] += deltaOculto[j]*lr;
                for(int k=0; k<numEntradas; k++) {
                    pesosOcultos[k][j]+=entrada_treinamento[i][k]*deltaOculto[j]*lr;
                }
            }
        }
    }

    for(int i=0; i<4; i++){
        std::cout << "Entrada:" << entrada_treinamento[i][0] << " " << entrada_treinamento[i][1] << "    Saída:" << camadaSaida[i] << "    Saída Esperada: " << saida_treinamento[i][0] << "\n";
    }

    // Print dos pesos
    std::cout << "\nPesos Ocultos Finais\n[ ";
    for (int j=0; j<numNosOcultos; j++) {
        std::cout << "[ ";
        for(int k=0; k<numEntradas; k++) {
            std::cout << pesosOcultos[k][j] << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";
    
    std::cout << "Bías Ocultos Finais\n[ ";
    for (int j=0; j<numNosOcultos; j++) {
        std::cout << camadasOcultasBias[j] << " ";

    }
    std::cout << "]\n";
    std::cout << "Pesos de Saída Finais";
    for (int j=0; j<numSaidas; j++) {
        std::cout << "[ ";
        for (int k=0; k<numNosOcultos; k++) {
            std::cout << pesosSaida[k][j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "Bías de Saída Finais\n[ ";
    for (int j=0; j<numSaidas; j++) {
        std::cout << camadaSaidaBias[j] << " ";
        
    }
    std::cout << "]\n";

    clock_t fim = clock();

    // calcula o tempo decorrido encontrando a diferença (fim - inicio) e
    // dividindo a diferença por CLOCKS_PER_SEC para converter em segundos
    tempo_gasto += (double)(fim - inicio) / CLOCKS_PER_SEC;
 
    std::cout << "\nTempo Gasto: " << tempo_gasto << " segundos";

    return 0;
}