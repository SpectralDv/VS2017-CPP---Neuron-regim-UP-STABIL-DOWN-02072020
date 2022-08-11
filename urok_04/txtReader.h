#pragma once

#include "lib.h"

vector<double> vAllWeight; //одномерный вектор всех весов

//загрузка данных из файла в одномерный вектор
void loadFile(string filename, int sizeAllWeight)
{
	vAllWeight.clear(); //очищает вектор
	vAllWeight.shrink_to_fit(); //удаляет пустые резервные места

	double vectorbuf; //буфер элементов веткора

	//считывает файл в поток
	ifstream ifile(filename);

	if (ifile.is_open())
	{
		cout << "Файл открыт. " << endl;
		while (ifile >> vectorbuf)
		{
			//записывает данные в вектор
			vAllWeight.push_back(vectorbuf);
		}
	}
	else
	{
		cout << "Не удалось открыть файл." << endl;
	}
	ifile.close();

	//cout << "\n\nКоличество весов: " << vAllWeight.size() << "\n\n";
}

//выгрузка данных из одномерного вектора в файл //vbuf - буфер обмена между векторами, sizeAllWeight - количество всех весов
void unloadFile(string filename, int sizeAllWeight) 
{
	//создает txt файл
	ofstream ofile(filename);

	vAllWeight.resize(sizeAllWeight); //резервирует места в векторе для всех весов

	//поочереди обрабатывает все веторы с весами
	for (int i = 0; i < vAllWeight.size(); i++)
	{
		//for (int j = 0; j < 1; j++)
		//{
			//отправляет элемент вектора весов в файл
			ofile << vAllWeight[i] << " ";
		//}
		//добавляет элемент в конец вектора
		//vWeight.push_back(vbuf);
		//заполняет нужный элемент нужным значением
		//vWeight[i] = vbuf;
	}
	ofile.close();
}

