#pragma once
#include <exception>
#include <string>
#include <random>

//能用一维数组就用一维数组，主要方便使用eigen库
//惊奇发现特么的用eigen库来后向传播速度慢得多
//10*10 -> 10*10 网络， 100000次传播，用eigen时耗时500+ms，不用时50ms
//后向传播还是用CBasicComputation中的函数，因为只涉及到了矩阵乘和加


static std::default_random_engine randomEngine;
//能量函数和激活函数单独写成了虚拟类和子类，方便后续扩充

//能量函数
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

//激活函数
//激活函数的Diff部分参数包括了激活前后的值，不一定能都用到（比如sigmoid值用激活后的），主要是更普适
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
	//本层网络节点数
	int nodeNum;
	//前一层网络节点数
	int nodeBef;

	//当前层中权重
	//数据排列方式按照 nodeBef * nodeNum 的结构
	double* weight;
	//最优时权重；这里理解有误，初始化时weight应该与weightBest一致（再非后向传播时），后向传播时应该更改weigtBest，完全传播后需要将weightBest赋给weight
	//为什么呢？因为后向传播每层需要后一层的weight值，这应该是未更改的
	double* weightBest;
	//各节点偏移
	double* bias;
	//这个同bias
	double* biasBest;
	//用于存储后向传播中激活函数的导数值，略去每次都有的new delete和memset操作
	double* diffActivationFun;
	//储存向前传播的临时量
	double* tempNodeValForeProgation;
	//向后传播的中间量
	double* tempXBackBackProgation;
	double* tempDWeightBackProgation;
	//用于存储后向传播中一行权重的修正值
	double* dWeight;
	//后向传播值
	double* xBack;
	//激活函数
	ActivationFun* activationFun;

	//是否完成初始化，否则不能进行传播
	bool isInitial;

	
public:	
	//层类型描述
	const enum LAYERTYPE {INPUT, HIDDEN, OUTPUT, VITUAL};
	LAYERTYPE layerType;
	//当前层中节点参数，经过激活函数激活后，未激活的部分在前、后向传播中都用不到
	double* nodeVal;
	//还是加上没激活的nodeVal，上面提到的后向传播里用不到其实只是对于sigmoid函数
	double* nodeValUnActived;
	//构造函
	BPLayer(int nodeNum, int nodeBef, LAYERTYPE layerType=HIDDEN, ActivationFun::ActivationFunType type= ActivationFun::ActivationFunType::SIGMOID);
	//层初始化
	bool Initial(std::default_random_engine e);
	bool Initial();
	//向前传播，注意此处必须传入对象指针，不然表面上是作为形参传入的，但是会在函数结束时调用析构
	//另外输出层经过sigmoid函数激活后会使输出值在(0, 1)需要恰当设置结果
	bool ForePropagate(const BPLayer* layerBef);
	//向后传播，输出层，隐藏层和输入层后向传播不一致，后两者通过层标识区分
	//对于非输出层的传播
	bool BackPropagate(const BPLayer* layerBef, const BPLayer* layerAft, double step);
	//对于输出层的传播，因为能量函数是Net的属性而不是Layer属性，这里就直接在开始传播时直接设置输出层的xBacks部分值，内部需要乘上 d (xActived) / d  (x)
	bool BackPropagate(const BPLayer* layerBef, double step);
	//设置激活函数
	bool SetActivationFun(ActivationFun::ActivationFunType type);
	//主要用于初始化输出层的向后传播值，其他层还是自己更改xBack值
	void SetTempXBack(double* xBack);
	//用weightBest,biasBest同步weight,bias，这个过程在整个网络完成了后向传播后进行
	void SynParameter();

	//发现析构函数不能自己调用，因为在离开变量作用域时会自己调用，此时会报错
	~BPLayer();

};

class BPNNet
{
	//输入层节点数
	int nodeNumInputLayer;
	//输出层节点数
	int nodeNumOutLayer;
	//隐藏层层数
	int layerNumHiddenLayer;
	//隐藏层节点数
	int* nodeNumHiddenLayer;

	//用于存储后向传播中能量函数的导数值，略去每次都有的new delete和memset操作
	double* diffEnergyFun;

	//步长
	//默认值
	const double STEP = 1;
	double step;

	//用链表形式存储网络，因为在先前传播过程里需要前一个节点的输出，向后传播需要前后两个节点的
	struct NetNode
	{
		BPLayer* layer;
		NetNode* nodeNext;
		NetNode* nodeBef;
	};
	//指向网络第一层，这里是一个不需要初始化的vitual层，主要是为了存输入值，简化网络传播中的操作
	NetNode* net;
	//指向输出层，便于拿出输出值，此外作为后向传播的起点
	NetNode* outputLayer;

	//能量函数
	EnergyFun* energyFun;
	//输出能量
	double energy;
	//是否完成初始化
	bool isInitial;

public:
	//public型输出值，向前传播后将输出层值赋给output
	double* output;
	BPNNet(int layerNum, int* nodeNums, EnergyFun::EnergyFunType type= EnergyFun::EnergyFunType::STD);
	~BPNNet();
	//设置各层的激活函数，会在构造函数中第一次设置，取决于BPLayer构造时的默认值(Sigmoid)
	bool SetActivationFun(int nLayer, ActivationFun::ActivationFunType activationFunType);
	bool SetActivationFun(ActivationFun::ActivationFunType* activationFunType);
	bool SetEnergyFun(EnergyFun::EnergyFunType type);
	bool Initial();
	void SetStep(double step);
	bool ForePropagate(double* input);
	bool BackPropagate(double* target);
	//训练网络
	bool Train(double* input, double* target, int dataSetSize, int cycle = 1);
	
};


//用来扔相关的异常
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

//基本矩阵计算函数
//点乘
void MatrixDot(double* mat, double x, int size, double* result);
void MatrixDot(double* mat1, double* mat2, int size, double* result);
void MatrixDot(double* mat, double* vector, int row, int col, bool isRowVec, double* result);
//加法
void MatrixAdd(double* mat1, double* mat2, int size, double* result);
//减法
void MatrixSub(double* mat1, double* mat2, int size, double* result);
//乘法
void MatrixMul(double* mat1, double* mat2, int row1, int col1_row2, int col2, double* result);


