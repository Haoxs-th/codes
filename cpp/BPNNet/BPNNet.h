#pragma once
#include <exception>
#include <string>
#include <random>

//����һά�������һά���飬��Ҫ����ʹ��eigen��
//���淢����ô����eigen�������򴫲��ٶ����ö�
//10*10 -> 10*10 ���磬 100000�δ�������eigenʱ��ʱ500+ms������ʱ50ms
//���򴫲�������CBasicComputation�еĺ�������Ϊֻ�漰���˾���˺ͼ�


static std::default_random_engine randomEngine;
//���������ͼ��������д��������������࣬�����������

//��������
class EnergyFun {
public:
	//CBasicComputation* BC;
	//EnergyFun():BC(new CBasicComputation){}
	const enum EnergyFunType{STD};
	virtual void Fun(double* x, double* target, int n, double* y) {}
	virtual void Diff(double* x, double* target, int n, double* diff) {}
};
class EnergyFunStd : public EnergyFun {
public:
	void Fun(double* x, double* target, int n, double* y);
	void Diff(double* x, double* target, int n, double* diff);
};

//�����
//�������Diff���ֲ��������˼���ǰ���ֵ����һ���ܶ��õ�������sigmoidֵ�ü����ģ�����Ҫ�Ǹ�����
class ActivationFun {
public:
	const enum ActivationFunType { SIGMOID };
	virtual void Active(double* x, double* xActived, int n = 1){}
	virtual void Diff(double* xActived, double* xUnActived, double* diff, int n = 1){}
};
class ActivationFunSigmoid : public ActivationFun {
public:
	virtual void Active(double* x, double* xActived, int n = 1);
	virtual void Diff(double* xActived, double* xUnActived, double* diff, int n = 1);
};

class BPLayer {
	//��������ڵ���
	int nodeNum;
	//ǰһ������ڵ���
	int nodeBef;

	//��ǰ����Ȩ��
	//�������з�ʽ���� nodeBef * nodeNum �Ľṹ
	double* weight;
	//����ʱȨ�أ�����������󣬳�ʼ��ʱweightӦ����weightBestһ�£��ٷǺ��򴫲�ʱ�������򴫲�ʱӦ�ø���weigtBest����ȫ��������Ҫ��weightBest����weight
	//Ϊʲô�أ���Ϊ���򴫲�ÿ����Ҫ��һ���weightֵ����Ӧ����δ���ĵ�
	double* weightBest;
	//���ڵ�ƫ��
	double* bias;
	//���ͬbias
	double* biasBest;
	//���ڴ洢���򴫲��м�����ĵ���ֵ����ȥÿ�ζ��е�new delete��memset����
	double* diffActivationFun;
	//������ǰ��������ʱ��
	double* tempNodeValForeProgation;
	//��󴫲����м���
	double* tempXBackBackProgation;
	double* tempDWeightBackProgation;
	//���ڴ洢���򴫲���һ��Ȩ�ص�����ֵ
	double* dWeight;
	//���򴫲�ֵ
	double* xBack;
	//�����
	ActivationFun* activationFun;

	//�Ƿ���ɳ�ʼ���������ܽ��д���
	bool isInitial;

	
public:	
	//����������
	const enum LAYERTYPE {INPUT, HIDDEN, OUTPUT, VITUAL};
	LAYERTYPE layerType;
	//��ǰ���нڵ��������������������δ����Ĳ�����ǰ�����򴫲��ж��ò���
	double* nodeVal;
	//���Ǽ���û�����nodeVal�������ᵽ�ĺ��򴫲����ò�����ʵֻ�Ƕ���sigmoid����
	double* nodeValUnActived;
	//���캯
	BPLayer(int nodeNum, int nodeBef, LAYERTYPE layerType=HIDDEN, ActivationFun::ActivationFunType type= ActivationFun::ActivationFunType::SIGMOID);
	//���ʼ��
	bool Initial(std::default_random_engine e);
	bool Initial();
	//��ǰ������ע��˴����봫�����ָ�룬��Ȼ����������Ϊ�βδ���ģ����ǻ��ں�������ʱ��������
	//��������㾭��sigmoid����������ʹ���ֵ��(0, 1)��Ҫǡ�����ý��
	bool ForePropagate(const BPLayer* layerBef);
	//��󴫲�������㣬���ز���������򴫲���һ�£�������ͨ�����ʶ����
	//���ڷ������Ĵ���
	bool BackPropagate(const BPLayer* layerBef, const BPLayer* layerAft, double step);
	//���������Ĵ�������Ϊ����������Net�����Զ�����Layer���ԣ������ֱ���ڿ�ʼ����ʱֱ������������xBacks����ֵ���ڲ���Ҫ���� d (xActived) / d  (x)
	bool BackPropagate(const BPLayer* layerBef, double step);
	//���ü����
	bool SetActivationFun(ActivationFun::ActivationFunType type);
	//��Ҫ���ڳ�ʼ����������󴫲�ֵ�������㻹���Լ�����xBackֵ
	void SetTempXBack(double* xBack);
	//��weightBest,biasBestͬ��weight,bias�����������������������˺��򴫲������
	void SynParameter();

	//�����������������Լ����ã���Ϊ���뿪����������ʱ���Լ����ã���ʱ�ᱨ��
	~BPLayer();

};

class BPNNet
{
	//�����ڵ���
	int nodeNumInputLayer;
	//�����ڵ���
	int nodeNumOutLayer;
	//���ز����
	int layerNumHiddenLayer;
	//���ز�ڵ���
	int* nodeNumHiddenLayer;

	//���ڴ洢���򴫲������������ĵ���ֵ����ȥÿ�ζ��е�new delete��memset����
	double* diffEnergyFun;

	//����
	//Ĭ��ֵ
	const double STEP = 1;
	double step;

	//��������ʽ�洢���磬��Ϊ����ǰ������������Ҫǰһ���ڵ���������󴫲���Ҫǰ�������ڵ��
	struct NetNode
	{
		BPLayer* layer;
		NetNode* nodeNext;
		NetNode* nodeBef;
	};
	//ָ�������һ�㣬������һ������Ҫ��ʼ����vitual�㣬��Ҫ��Ϊ�˴�����ֵ�������紫���еĲ���
	NetNode* net;
	//ָ������㣬�����ó����ֵ��������Ϊ���򴫲������
	NetNode* outputLayer;

	//��������
	EnergyFun* energyFun;
	//�������
	double energy;
	//�Ƿ���ɳ�ʼ��
	bool isInitial;

public:
	//public�����ֵ����ǰ�����������ֵ����output
	double* output;
	BPNNet(int layerNum, int* nodeNums, EnergyFun::EnergyFunType type= EnergyFun::EnergyFunType::STD);
	~BPNNet();
	//���ø���ļ���������ڹ��캯���е�һ�����ã�ȡ����BPLayer����ʱ��Ĭ��ֵ(Sigmoid)
	bool SetActivationFun(int nLayer, ActivationFun::ActivationFunType activationFunType);
	bool SetActivationFun(ActivationFun::ActivationFunType* activationFunType);
	bool SetEnergyFun(EnergyFun::EnergyFunType type);
	bool Initial();
	void SetStep(double step);
	bool ForePropagate(double* input);
	bool BackPropagate(double* target);
	//ѵ������
	bool Train(double* input, double* target, int dataSetSize, int cycle = 1);
	
};


//��������ص��쳣
class BPNNetException : public std::exception
{
public:
	BPNNetException(): message("BPNNet Error."){}
	BPNNetException(std::string str) : message("BPNNet Error: " + str){}

	virtual const char* what() const throw () {
		return message.c_str();
	}

private:
	std::string message;
};

//����������㺯��
//���
void MatrixDot(double* mat, double x, int size, double* result);
void MatrixDot(double* mat1, double* mat2, int size, double* result);
void MatrixDot(double* mat, double* vector, int row, int col, bool isRowVec, double* result);
//�ӷ�
void MatrixAdd(double* mat1, double* mat2, int size, double* result);
//����
void MatrixSub(double* mat1, double* mat2, int size, double* result);
//�˷�
void MatrixMul(double* mat1, double* mat2, int row1, int col1_row2, int col2, double* result);


