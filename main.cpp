#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ceres/ceres.h"
#include <chrono>

using namespace std;
using namespace ceres;
using namespace cv;

#define fx 3
#define fy 2
#define cx 2
#define cy 1
struct CostFunctor{
    template <typename T>
    bool operator()(const T* const x, T* residual)const{
        residual[0] = T(10.0) - x;
        return true;
    }
};


class Rat_Ana:public SizedCostFunction<1,4>{
public:
    Rat_Ana(const double x, const double y) : x_(x), y_(y) {}
    virtual ~Rat_Ana() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double b1 = parameters[0][0];
        const double b2 = parameters[0][1];
        const double b3 = parameters[0][2];
        const double b4 = parameters[0][3];

        residuals[0] = b1 *  pow(1 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;

        if (!jacobians) return true;
        double* jacobian = jacobians[0];
        if (!jacobian) return true;

        jacobian[0] = pow(1 + exp(b2 - b3 * x_), -1.0 / b4);
        jacobian[1] = -b1 * exp(b2 - b3 * x_) *
                      pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
        jacobian[2] = x_ * b1 * exp(b2 - b3 * x_) *
                      pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
        jacobian[3] = b1 * log(1 + exp(b2 - b3 * x_)) *
                      pow(1 + exp(b2 - b3 * x_), -1.0 / b4) / (b4 * b4);
        return true;
    }

private:
    const double x_;
    const double y_;
};

class analyticalmethod: public SizedCostFunction<1,3>{
    //construction
    analyticalmethod(double paras){}
    //desconstruction
    virtual ~analyticalmethod(){}
    //overload evaluate
    virtual bool Evaluate(double const* const* para,
            double* residual,
            double** jacobians)const{
        residual[0];

        if(!jacobians && !jacobians[0])
            double jacobian ;
        return true;
    }
};

struct autodiffmethod{
    //construction
    //overload operator()
    template <typename T>
    bool operator()(const T* const paras, T* residual) const {

        residual[0];
        return true;
    }
    //structure to hide the ???构造CostFunction，隐藏内部结构
    CostFunction* Create(double para){
        return(
                new AutoDiffCostFunction<autodiffmethod,1,3>(
                        new autodiffmethod()
                        )
                );
    }
};

class BA_analytical:public SizedCostFunction<2,6>{
public:
    BA_analytical(Eigen::Vector2d observed_p):_observed_p(observed_p) {}
    virtual ~BA_analytical(){}
    virtual bool Evaluate(double const* const* parameters,
            double* residuals,
            double** jacobians)const{
        //parameters中R用四元数传，顺序为：Pwx, Pwy, Pwz, q1,q2,q3,qw, t1, t2, t3
        //先用eigen写，Sophus以后再改进吧。目前还不太会用sophus
        //四元数转旋转矩阵：
        Eigen::Vector3d Pw, Pc, t;
        Pw<<parameters[0][0], parameters[0][1], parameters[0][2];
        t<<parameters[2][0], parameters[2][1], parameters[2][2];
        Eigen::Quaterniond q(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Matrix3d R = q.toRotationMatrix();
        Pc =  R*Pw+t;
        double X = Pc(0);
        double Y = Pc(1);
        double Z = Pc(2);
        double X2 = X*X;
        double Y2 = Y*Y;
        double Z2 = Z*Z;

        residuals[0] = _observed_p(0) - cx - fx*X/Z;
        residuals[1] = _observed_p(1) - cy - fy*Y/Z;

        if(!jacobians) return true; //为什么要保证jacobians不为NULL且jacobians[0]不为NULL？
        double* jacobian = jacobians[0];
        if(!jacobian) return true;
        jacobian[0] = fx*X*Y/Z2;
        jacobian[1] = -fx - fx*X2/Z2;
        jacobian[2] = fx*Y/Z;
        jacobian[3] = -fx/Z;
        jacobian[4] = 0;
        jacobian[5] = X2/Z2;

        jacobian[6] = fy + fy*Y2/Z2;
        jacobian[7] = -fy*X*Y/Z2;
        jacobian[8] = -fy*X/Z;
        jacobian[9] = 0;
        jacobian[10] = -fy/Z;
        jacobian[11] = fy*Y/Z2;
        return true;
    }
private:
    Eigen::Vector2d _observed_p;
};

class Rat_Ana_Optimal:public SizedCostFunction<1,4>{
public:
    Rat_Ana_Optimal(double x, double y):_x(x), _y(y){}
    virtual ~Rat_Ana_Optimal(){}
    virtual bool Evaluate(double const* const* parameters,
            double* residual,
            double** jacobians)const{
        double b1 = parameters[0][0];
        double b2 = parameters[0][1];
        double b3 = parameters[0][2];
        double b4 = parameters[0][3];

        double t1 = exp(b2-b3*_x);
        double t2 = 1+t1;
        double t3 = pow(t2, -1.0/b4);
        double t4 = pow(t2, -1.0/b4-1); //难道是因为这个放错了地方？

        residual[0] = -_y + b1*t3;          //这里的正负会影响结果？

        if(!jacobians) return true;
        double* jacobian = jacobians[0];
        if(!jacobian) return true;

        jacobian[0] = t3;
        jacobian[1] = -b1*t1*t4/b4;
        jacobian[2] = -_x*jacobian[1];
        jacobian[3] = b1*log(t2)*t3/(b4*b4);

        return true;
    }
private:
    double _x;
    double _y;
};


struct Rat_Autodiff{
    Rat_Autodiff(double x, double y):_x(x), _y(y){}
    template <typename T>
    bool operator()(const T* para, T* residual)const{
        T b1 = para[0];
        T b2 = para[1];
        T b3 = para[2];
        T b4 = para[3];

        residual[0] = b1*pow(T(1)+exp(b2-b3*_x), -T(1)/b4) - _y;    //数字都要转换成T()格式
        return true;                        //这句话也必须有，但不知道为啥
    }

    static CostFunction* Create(double x, double y){
        return (
                new AutoDiffCostFunction<Rat_Autodiff, 1, 4>(
                        new Rat_Autodiff(x,y)
                        )
                );
    }
private:
    double _x;
    double _y;
};

struct Rat_Numeridiff{
    Rat_Numeridiff(double x, double y):_x(x), _y(y){}
    template <typename T>
    bool operator()(const T* const para, T* residual)const{     //这里也一定要有const，否则报错balabala自己看！
        T b1 = para[0];
        T b2 = para[1];
        T b3 = para[2];
        T b4 = para[3];

        residual[0] = b1*pow(T(1) + exp(b2-b3*_x), -T(1)/b4);
        return true;
    }
    static CostFunction* Create(const double x, const double y){
        return (
                new NumericDiffCostFunction<Rat_Numeridiff, RIDDERS, 1, 4>(         //可选有：FORWARD， CENTRAL， RIDDERS，效果都不好啊，是这次编的有问题呢还是本身计算效果就不好?
                        new Rat_Numeridiff(x, y)
                        )
                );
    }
private:
    double _x;
    double _y;
};
struct curvefit{
    curvefit(double x, double y):_x(x),_y(y){}

    template <typename T>
    bool operator()(const T* const b, T* residual)const{    //注意这里的const符号，T参数类型及operator()(***)const
        residual[0] = _y - b[0]*exp(b[1]*_x*_x+b[2]*_x+b[3]);
        return true;
    }
    static CostFunction* Create(const double x, const double y){
        //return应该接()而不是{}
        return(
            new AutoDiffCostFunction<curvefit, 1, 4>(
                    new curvefit(x,y)
                    )
        );
    }
private:
    const double _x;
    const double _y;
};

struct curve_fitting{
    curve_fitting(double x, double y):_x(x), _y(y){}
    template <typename T>
    bool operator()(const T* const para, T* residual)const {
        residual[0] = exp(para[0]*_x*_x+para[1]*_x+para[2])-_y;
        return true;
    }

    static ceres::CostFunction* Create(const double x, const double y){
        return(
                new ceres::AutoDiffCostFunction<curve_fitting, 1, 3>(
                        new curve_fitting(x, y)
                )
        );
    }
private:
    double _x;
    double _y;
};

//class test : public FirstOrderFunction

class curve_fitting_ana:public SizedCostFunction<1,3>{
public:
    curve_fitting_ana(double x, double y): _x(x), _y(y){}
    virtual ~curve_fitting_ana(){}
    virtual bool Evaluate(double const* const* para,
            double* residual,
            double** jacobians)const{

        const double a = para[0][0];
        const double b = para[0][1];
        const double c = para[0][2];

        residual[0] = _y - exp(a*_x*_x+b*_x+c);

        if(!jacobians) return true;
        double* jacobian = jacobians[0];
        if(!jacobian) return true;

        jacobian[0] = _x*_x*residual[0];
        jacobian[1] = _x*residual[0];
        jacobian[2] = residual[0];

        return true;
    }
private:
    double _x, _y;
};

class cfa_5para:public SizedCostFunction<1,5>{
public:
    cfa_5para(double x, double y): _x(x), _y(y){}
    virtual ~cfa_5para(){}
    virtual bool Evaluate(double const* const* para,
            double* residual,
            double** jacobians)const{       //这里也不要忘记const，这是表达的啥含义啊？
        double a = para[0][0];
        double b = para[0][1];
        double c = para[0][2];
        double d = para[0][3];
        double e = para[0][4];

        double common = exp(b*_x*_x+c*_x+d);

        residual[0] = _y-(a*common-e/_x);

        if(!jacobians) return true;
        double* jacobian = jacobians[0];
        if(!jacobian) return true;

        jacobian[0] = common;
        jacobian[1] = a*_x*_x*common;
        jacobian[2] = a*_x*common;
        jacobian[3] = a*common;
        jacobian[4] = -1/_x;
        return true;
    }
public:
    double _x;
    double _y;
};
int main() {
    std::cout << "Hello, ceres World!" << std::endl;
    double b[4] = {16.3, 2.08, 0.1956, 0.687};
    Problem problem;

    /*double initial_x = 5.0;
    double x = initial_x;
    CostFuncton* costFuncton = new AutoDiffCostj
              nnFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(costFuncton,NULL, &x);*/

    ofstream InitialPoints;
    InitialPoints.open("InitialPoints", ios::binary |ios::trunc|ios::in|ios::out);
    RNG rng;
    vector<Point2d> p2dv;
    double cf_b[3] = {1.0, 2.0, 1.0};
    double cf_4ana[4] = {699, 5.27, 0.759, 1.279};
    double cfa_para[5] = {3,1,2,1,3};
    for(int i = 0; i<100; i++){
        //double x = data[2*i+1];
        //double y = data[2*i];
        double x = i;
        //double y = cf_b[0]*exp(cf_b[1]*x*x+cf_b[2]*x+cf_b[3])+rng.gaussian(0.2*0.2);
        //double y = exp(cf_b[0]*x*x+cf_b[1]*x+cf_b[2])+rng.gaussian(1*1);
        //double y = cfa_para[0]*exp(cfa_para[1]*x*x+cfa_para[2]*x+cfa_para[3])-cfa_para[4]/x;
        double y = cf_4ana[0]*pow(1 + exp(cf_4ana[1] -  cf_4ana[2] * x), -1.0 / cf_4ana[3])+rng.gaussian(2*2);
        p2dv.push_back(Point2d(x, y));
    }

    double b_esti[4] = {100, 10, 1, 1};
    double cf_b_esti[3] = {2, -1, 5};
    double cfa_para_esti[5] = {1, 1, 1, 1, 2 };
    for(int i = 0; i<100; i++){
        //怎么用那个CostFunctor？？
        double x = p2dv[i].x;
        double y = p2dv[i].y;
        cout<<"x = "<<x<<"\ty = "<<y<<endl;
        InitialPoints<<x<<" "<<y<<endl;
        //CostFunction* cost_function = new CostFunctor_ana(x, y);
        //CostFunction* cost_function = curve_fitting::Create(x, y);   //这句话写得不对，or create函数写得不对！发现了，是函数名字写错了
        //CostFunction* cost_function = new AutoDiffCostFunction<curve_fitting, 1, 3>(new curve_fitting(x, y));
        //CostFunction* cost_function = new curve_fitting_ana(x,y);
        if(x != 0 && y != 0){       //注意排除不可求导的情况
            //CostFunction* cost_function = new cfa_5para(x,y);
            //CostFunction* cost_function = Rat_Numeridiff::Create(x, y);
            CostFunction* cost_function = new BA_analytical(Eigen::Vector2d(x,y));
            problem.AddResidualBlock(cost_function, NULL, b_esti);
        }
        else continue;
    }
    //是不是没有evaluate，不是，是jacobian的值nan/inf

    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1.0e-20;
    options.parameter_tolerance = 1.0e-20;
    Solver::Summary summary;
    chrono::steady_clock::time_point t_begin = chrono::steady_clock::now();
    Solve(options, &problem, &summary); //这句话就是在求解
    chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
    cout<<summary.FullReport()<<endl;
    //cout<<"curve fitting parameter: a, b, c= "<<cfa_para_esti[0]<<"\t"<<cfa_para_esti[1]<<"\t"<<cfa_para_esti[2]<<cfa_para_esti[3]<<cfa_para_esti[4]<<"\t"<<endl;
    chrono::duration<double> time_cost = chrono::duration_cast<chrono::duration<double>>(t_end-t_begin);
    cout<<"curve fitting parameter: a, b, c, d= "<<b_esti[0]<<"\t"<<b_esti[1]<<"\t"<<b_esti[2]<<"\t"<<b_esti[3]<<endl;
    cout<<"Auto differential ceres time-cost is: "<<time_cost.count()<<endl;

    ofstream EstimatedPoints;
    EstimatedPoints.open("EstimatedPoints", ios::binary |ios::trunc|ios::in|ios::out);
    for(int i = 0; i<100; i++){
        double x = i;
        //double x = data[2*i+1];
        //double y = cf_b[0]*exp(cf_b[1]*x*x+cf_b[2]*x+cf_b[3])+rng.gaussian(0.2*0.2);
        //double y = exp(cf_b[0]*x*x+cf_b[1]*x+cf_b[2])+rng.gaussian(1*1);
        double y = b_esti[0]*pow(1+exp(b_esti[1]-b_esti[2]*x),-1/b_esti[3]);
        EstimatedPoints<<x<<" "<<y<<endl;
    }

    return 0;
}