#include <iostream>
#include <fstream>
#ifdef _UNIX
#include <cmath>
#elif defined _WIN32
#include <corecrt_math_defines.h>
#endif
#include <vector>
#include <string>

using namespace std;

struct Complex{
    float re{0.f};
    float im{0.f};

    Complex & operator+(const Complex &y) {
        this->re += y.re;
        this->im += y.im;
        return *this;
    }

    Complex &operator-(const Complex &y) {
        this->re -= y.re;
        this->im -= y.im;
        return *this;
    }

    Complex &operator*(const Complex &y) {
        this->re = this->re * y.re - this->im * y.im;
        this->im = this->re * y.im + this->im * y.re;
        return *this;
    }

    static Complex add(const Complex &x, const Complex &y)
    {
        Complex w;
        w.re = x.re + y.re;
        w.im = x.im + y.im;
        return w;
    }

    static Complex sub(const Complex &x, const Complex &y) {
        Complex w;
        w.re = x.re - y.re;
        w.im = x.im - y.im;
        return w;
    }

    static Complex mul(const Complex &x, const Complex &y) {
        Complex w;
        w.re = x.re * y.re - x.im * y.im;
        w.im = x.re * y.im + x.im * y.re;
        return w;
    }

    friend ostream & operator<<(ostream &out, const Complex &x) {
        out << x.re << "+j" << x.im;
        return out;
    }
};

/**
 * @brief 
 * 
 *
 * @param x 
 * @return 
 */
vector<Complex> fft(const vector<Complex>& x) {
    auto n = x.size();
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("The length of time domain sequence must be a power of 2.");
    }
    else if (n == 1) {
        return x;
    }
    vector<Complex> w(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        float angle = -i * 2.f * M_PI / (float)n;
        w[i].re = cos(angle);
        w[i].im = sin(angle);
    }
    vector<Complex> x_0(n / 2);
    vector<Complex> x_1(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        x_0[i] = x[2 * i];
        x_1[i] = x[2 * i + 1];
    }
    auto y = fft(x_0);
    auto z = fft(x_1);
    
    vector<Complex> result(n); 
    for (int i = 0; i < n / 2; ++i) {
        result[i] = y[i] + w[i] * z[i];
        result[i + n / 2] = y[i] - w[i] * z[i];
    }
    return result;
}

istream & operator>>(istream &in, vector<float> &x) {
    string c;
    float num = 0.f;
    int flag = 0;
    char last = '0';
    while ((c = cin.get()).data()[0] != '\n') {
        if (c.data()[0] == ' ' && last == ' ') {
            last = ' ';
            continue;
        }
        else if (c.data()[0] == ' ') {
            x.push_back(num);
            flag = 0;
            num = 0.f;
        }
        else if (c.data()[0] == '.') {
            flag = -1;
        }
        else if (c.data()[0] < '0' || c.data()[0] > '9') {
            exit(1);
        }
        else {
            if (flag == 0) {
                num *= 10.f;
                num += (float)(stoi(c));
            }
            else if (flag < 0) {
                num += pow(10.f, flag) * (float)(stoi(c));
                flag -= 1;
            }
        }
        last = c.data()[0];
    }
    x.push_back(num);
    num = 0.f;

    return in;
}

int main(int argc, char *argv) {
    vector<float> x;
    cin >> x;
    vector<Complex> c;

    for (auto &num : x) {
        Complex complex;
        complex.re = num;
        cout << complex << ' ';
        c.push_back(complex);
    }
    cout << '\n';
    
    vector<Complex> result;
    try {
        result = fft(c);
    } catch (std::invalid_argument &ex) {
        cerr << "Invalid Argument: " << ex.what() << '\n';
    }
    for (auto &num : result) {
        cout << num << ' ';
    }
    cout << '\n';
    return 0;
}