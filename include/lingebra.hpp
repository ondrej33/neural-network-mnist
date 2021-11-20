#include <vector>
#include <utility>
#include <cassert>
#include <iomanip>
#include <iostream>


/**
 * Vector of doubles with all necessary operators defined 
 */

class DoubleVec
{
    std::vector<double> _vec;

public:
    DoubleVec(): _vec() {}

    /* construct a zero DoubleVec of given dimension */
    explicit DoubleVec(int dimension): _vec(std::vector<double>(dimension)) {}

    /* construct a DoubleVec from given vector of doubles */
    explicit DoubleVec(std::vector<double> &v): _vec(v) {}
    explicit DoubleVec(std::vector<double> &&v): _vec(std::move(v)) {}

    int size() const { return _vec.size(); }
    bool empty() const { return _vec.empty(); }
    void push_back(double num) { _vec.push_back(num); }
    void pop_back(double num) { _vec.pop_back(); }

    friend DoubleVec operator-(DoubleVec vec);
    friend DoubleVec operator+(DoubleVec first, const DoubleVec& second);
    friend DoubleVec operator-(DoubleVec first, const DoubleVec& second);
    friend double operator*(const DoubleVec& first, const DoubleVec& second);

    friend DoubleVec operator*(DoubleVec vec, double scalar);
    friend DoubleVec operator*(double scalar, DoubleVec vec);
    friend DoubleVec operator/(DoubleVec vec, double scalar);
    friend DoubleVec operator/(double scalar, DoubleVec vec);

    DoubleVec& operator+=(const DoubleVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); i++) {
            _vec[i] += other._vec[i];
        }
        return *this;
    }

    DoubleVec& operator-=(const DoubleVec& other)
    {
        this->compareSizes(other);
        for (int i = 0; i < size(); i++) {
            _vec[i] -= other._vec[i];
        }
        return *this;
    }

    DoubleVec& operator*=(double scalar)
    {
        for (int i = 0; i < size(); i++) {
            _vec[i] *= scalar;
        }
        return *this;
    }

    // having *= for dot product does not make sense

    DoubleVec& operator/=(double scalar)
    {
        for (int i = 0; i < size(); i++) {
            _vec[i] /= scalar;
        }
        return *this;
    }

    bool operator==(const DoubleVec& other) const { return this->_vec == other._vec; }

    bool operator!=(const DoubleVec& other) const { return !(*this == other); }

    double operator[](int i) const 
    { 
        assert(i >= 0 && i <= size() - 1);
        return _vec[i]; 
    }

    double& operator[](int i) 
    { 
        assert(i >= 0 && i <= size() - 1);
        return _vec[i]; 
    }

    std::vector<double>::iterator begin() { return _vec.begin(); }
    std::vector<double>::iterator end() { return _vec.end(); }

    std::vector<double>::const_iterator begin() const { return _vec.begin(); }
    std::vector<double>::const_iterator end() const { return _vec.end(); }

    void compareSizes(const DoubleVec& other) const;
};

/* unary minus */
DoubleVec operator-(DoubleVec vec)
{   
    for (int i = 0; i < vec._vec.size(); i++) {
        vec._vec[i] *= -1;
    }
    return vec;
}

/* matrix addition and subtraction */
DoubleVec operator+(DoubleVec first, const DoubleVec& second)
{ return first += second; }

DoubleVec operator-(DoubleVec first, const DoubleVec& second)
{ return first -= second; }

/* dot product */
double operator*(const DoubleVec& first, const DoubleVec& second)
{
    first.compareSizes(second);
    double result = 0;
    for (int i = 0; i < first._vec.size(); i++) {
        result += first._vec[i] * second._vec[i];
    }
    return result;
}

/* Multiplication and division by scalar */
DoubleVec operator*(DoubleVec vec, double scalar)
{ return vec *= scalar; }

DoubleVec operator*(double scalar, DoubleVec vec)
{ return vec *= scalar; }

DoubleVec operator/(DoubleVec vec, double scalar)
{ return vec /= scalar; }

DoubleVec operator/(double scalar, DoubleVec vec)
{ return vec /= scalar; }


/**
 * @brief Check if the another vector is of the same length as ours
 *      Can be removed during production, just make this function empty and
 *      compiler will do the rest (Preferably by DEBUG macro )
 * @param other 
 */
void DoubleVec::compareSizes(const DoubleVec& other) const {  
    #ifdef DEBUG
    if (size() != other.size()) {
        throw std::length_error("Vectors have different length");
    }    
    #endif
}

/**
 * Matrix of doubles with all necessary operators defined 
 */

class DoubleMat
{
    std::vector<DoubleVec> _matrix_rows;
public:
    /* construct a zero matrix */
    explicit DoubleMat(int rows, int columns) : _matrix_rows(rows, DoubleVec(columns)) {}

    /* construct a DoubleMat from given vectors of DoubleVec */
    explicit DoubleMat(const std::vector<DoubleVec> &rows) : _matrix_rows(rows) {}
    explicit DoubleMat(const std::vector<DoubleVec> &&rows) : _matrix_rows(std::move(rows)) {}

    int row_num() const
    { return _matrix_rows.size(); }

    int col_num() const
    { return (row_num() == 0) ? 0 : _matrix_rows[0].size(); }

    bool same_dimensions(const DoubleMat &other) const
    { return col_num() == other.col_num() && row_num() == other.row_num(); }

    /* Direct access to the COPY of column and row vectors
     * The ‹n› is index from top (row) or left (column) */

    DoubleVec row( int n ) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n];
    }

    DoubleVec col( int n ) const
    {
        assert(n >= 0 && col_num() >= n + 1);

        DoubleVec column(row_num());
        for (int i = 0; i < row_num(); i++) {
            column[i] = _matrix_rows[i][n];
        }
        return column;
    }

    /* relational operators */
    bool operator==(const DoubleMat &other) const
    { return _matrix_rows == other._matrix_rows; }

    bool operator!=(const DoubleMat &other) const
    { return _matrix_rows != other._matrix_rows; }

    /* constant / non-constant indexing to get or change the row vectors */
    DoubleVec operator[](int n) const
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    DoubleVec &operator[](int n)
    { 
        assert(n >= 0 && n <= row_num() - 1);
        return _matrix_rows[n]; 
    }

    /* matrix addition and subtraction */
    DoubleMat &operator+=(const DoubleMat &other)
    {        
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); i++) {
            _matrix_rows[i] += other._matrix_rows[i];
        }
        return *this;
    }

    DoubleMat &operator-=(const DoubleMat &other)
    {
        assert(same_dimensions(other));
        for (int i = 0; i < row_num(); i++) {
            _matrix_rows[i] -= other._matrix_rows[i];
        }
        return *this;
    }

    /* multiplying by a scalar */
    DoubleMat &operator*=(const double &scal)
    {
        for (int i = 0; i < row_num(); i++) {
            _matrix_rows[i] *= scal;
        }
        return *this;
    }

    // having *= for matrix product does not make sense

    /* get a transpose of the matrix */
    DoubleMat transpose() const
    {
        auto transp = DoubleMat(col_num(), row_num());
        for (int i = 0; i < col_num(); i++) {
            transp[i] = col(i);
        }
        return transp;
    }

    /* iterators */
    std::vector<DoubleVec>::iterator begin() { return _matrix_rows.begin(); }
    std::vector<DoubleVec>::iterator end() { return _matrix_rows.end(); }

    std::vector<DoubleVec>::const_iterator begin() const { return _matrix_rows.begin(); }
    std::vector<DoubleVec>::const_iterator end() const { return _matrix_rows.end(); }

    /* adds given vector to all rows */
    DoubleMat &add_vec_to_all_rows(const DoubleVec &vec)
    {        
        assert(vec.size() == col_num());
        for (int i = 0; i < row_num(); i++) {
            _matrix_rows[i] += vec;
        }
        return *this;
    }

};


/* matrix addition and subtraction */
DoubleMat operator+(DoubleMat x, const DoubleMat &y)
{ return x += y; }

DoubleMat operator-(DoubleMat x, const DoubleMat &y)
{ return x -= y; }

/* multiplication of a matrix by a scalar - both sides */
DoubleMat operator*(DoubleMat m, double scalar)
{ return m *= scalar; }

DoubleMat operator*(double scalar, DoubleMat m)
{ return m *= scalar; }

/* multiplication of a vector by a matrix - both sides 
 * if vector is first param, it is considered as [1, n] matrix
 * if vector is second param, it is considered as [n,1] matrix */
DoubleVec operator*(DoubleVec v, const DoubleMat &m)
{ 
    assert(m.row_num() == v.size());
    auto result = DoubleVec(m.col_num());

    for (int i = 0; i < m.col_num(); i++) {
        result[i] = v * m.col(i);
    }
    return result;
}

DoubleVec operator*(const DoubleMat &m, DoubleVec v)
{ 
    assert(m.col_num() == v.size());
    auto result = DoubleVec(m.row_num());

    for (int i = 0; i < m.row_num(); i++) {
        result[i] = m.row(i) * v;
    }
    return result;
}

/* multiplication of compatible matrices */
DoubleMat operator*(const DoubleMat &first, const DoubleMat &second)
{ 
    assert(first.col_num() == second.row_num());
    auto result = DoubleMat(first.row_num(), second.col_num());

    // TODO - check if ok
    for (int i = 0; i < first.row_num(); i++) {
        for (int j = 0; j < second.col_num(); j++) {
            result[i][j] = first.row(i) * second.col(j);
        }
    }
    return result;
}

/* printing a matrix, double printed to 4 decimal places */
inline void print_matrix(const DoubleMat &m)
{
    std::cout << std::setprecision(4) << std::fixed;

    for (int i = 0; i < m.row_num(); i++) {
       for (int j = 0; j < m.col_num(); j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}
