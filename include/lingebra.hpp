#include <vector>
#include <utility>


/**
 * Vector of doubles with all necessary operators defined 
 */

class DoubleVec
{
    std::vector<double> _vec;

public:
    DoubleVec(): _vec() {}

    DoubleVec(int size): _vec(std::vector<double>(size)) {}

    DoubleVec(std::vector<double> v): _vec(std::move(v)) {}

    int size() const { return _vec.size(); }
    bool empty() const { return _vec.empty(); }

    friend DoubleVec operator-(DoubleVec vec);
    friend DoubleVec operator+(DoubleVec first, const DoubleVec& second);
    friend DoubleVec operator-(DoubleVec first, const DoubleVec& second);
    friend double operator*(const DoubleVec& first, const DoubleVec& second);

    friend DoubleVec operator*(DoubleVec vec, double scalar);
    friend DoubleVec operator/(DoubleVec vec, double scalar);

    DoubleVec& operator+=(const DoubleVec& other)
    {
        *this = *this + other;
        return *this;
    }

    DoubleVec& operator-=(const DoubleVec& other)
    {
        *this = *this - other;
        return *this;
    }

    DoubleVec& operator*=(double scalar)
    {
        *this = *this * scalar;
        return *this;
    }

    DoubleVec& operator/=(double scalar)
    {
        *this = *this / scalar;
        return *this;
    }

    bool operator==(const DoubleVec& other) const { return this->_vec == other._vec; }

    bool operator!=(const DoubleVec& other) const { return !(*this == other); }

    double operator[](int i) const { return _vec[i]; }

    double& operator[](int i) { return _vec[i]; }

    std::vector<double>::iterator begin() { return _vec.begin(); }
    std::vector<double>::iterator end() { return _vec.end(); }

    std::vector<double>::const_iterator begin() const { return _vec.begin(); }
    std::vector<double>::const_iterator end() const { return _vec.end(); }

};

DoubleVec operator-(DoubleVec vec)
{
    for (int i = 0; i < vec._vec.size(); i++) {
        vec._vec[i] *= -1;
    }
    return vec;
}

DoubleVec operator+(DoubleVec first, const DoubleVec& second)
{
    for (int i = 0; i < first._vec.size(); i++) {
        first._vec[i] += second._vec[i];
    }
    return first;
}

DoubleVec operator-(DoubleVec first, const DoubleVec& second)
{
    for (int i = 0; i < first._vec.size(); i++) {
        first._vec[i] -= second._vec[i];
    }
    return first;
}

double operator*(const DoubleVec& first, const DoubleVec& second)
{
    double result = 0;
    for (int i = 0; i < first._vec.size(); i++) {
        result += first._vec[i] * second._vec[i];
    }
    return result;
}

DoubleVec operator*(DoubleVec vec, double scalar)
{
    for (int i = 0; i < vec._vec.size(); i++) {
        vec._vec[i] *= scalar;
    }
    return vec;
}

DoubleVec operator/(DoubleVec vec, double scalar)
{
    for (int i = 0; i < vec._vec.size(); i++) {
        vec._vec[i] /= scalar;
    }
    return vec;
}