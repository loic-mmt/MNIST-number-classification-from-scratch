#include <armadillo>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <chrono>
//#include <iterator>
#include <fstream>
//#include <sstream>
#include <locale>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

// Sigmoid activation (element-wise)
static arma::mat sigmoid(const arma::mat& x) {
    return 1.0 / (1.0 + arma::exp(-x));
}

// Derivative of sigmoid when given sigmoid(x) as input (element-wise)
static arma::mat sigmoid_deriv(const arma::mat& a) {
    return a % (1.0 - a);
}

static arma::mat softmax(const arma::mat& z) {
    arma::mat shifted = z.each_col() - arma::max(z, 1);
    arma::mat e = arma::exp(shifted);
    arma::colvec denom = arma::sum(e, 1);
    return e.each_col() / denom;
}

static arma::rowvec one_hot(std::size_t label, std::size_t k = 10){
    if (label >= k) {
        throw std::invalid_argument("label must be in [0, k)");
    }

    arma::rowvec Y(k, arma::fill::zeros);
    Y(label) = 1.0;
    return Y;
}


static arma::mat dropout_mask(arma::uword n_rows, arma::uword n_cols, double keep_prob) {
    if (keep_prob <= 0.0 || keep_prob > 1.0) {
        throw std::invalid_argument("keep_prob must be in (0, 1]");
    }

    arma::mat mask = arma::randu<arma::mat>(n_rows, n_cols);
    // for every value v:
    // if v < keep_prob -> syn kept (value 1/keep_prob)
    // if v > keep_prob -> syn supressed (value 0)
    mask.transform([keep_prob](double v) {
        return (v < keep_prob) ? (1.0 / keep_prob) : 0.0;
    });
    return mask;
}

class CSVRow
{
    public:
        std::string_view operator[](std::size_t index) const
        {
            return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
        }
        std::size_t size() const
        {
            return m_data.size() - 1;
        }
        void readNextRow(std::istream& str)
        {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find(',', pos)) != std::string::npos)
            {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
} 

struct PixelCol {
    int r;
    int c;
    std::size_t idx;
};

struct MnistData {
    arma::mat X;
    arma::uvec y;
    std::size_t n = 0;
    std::size_t n_features = 0;
};

struct MnistTestData {
    arma::mat X;
    std::vector<std::string> ids;
    std::size_t n = 0;
    std::size_t n_features = 0;
};

struct NNModel {
    arma::mat W0, W1;
    arma::rowvec b0, b1;
    double best_acc = -1.0;
};

static std::ifstream open_mnist_file() {
    // Shape: (30000, 787)
    std::ifstream file("data/train_mnist.csv");
    if (!file) {
        std::cerr << "Cannot open file\n";
    }
    return file;
}

static std::ifstream open_test_mnist_file() {
    // Shape: (40000, 786)
    std::ifstream file("data/test_mnist.csv");
    if (!file) {
        std::cerr << "Cannot open file\n";
    }
    return file;
}

static bool read_header(std::ifstream& file, CSVRow& row) {
    // --- Read header ---
    if (!(file >> row)) {
        std::cerr << "Empty file\n";
        return false;
    }
    return true;
}

static bool parse_header_columns(
    const CSVRow& row,
    std::size_t& label_idx,
    std::vector<PixelCol>& pixel_cols,
    std::size_t& id_idx,
    bool has_label
) {
    // Locate label/id column and pixel columns like in the Python script
    label_idx = static_cast<std::size_t>(-1);
    id_idx = static_cast<std::size_t>(-1);
    pixel_cols.clear();
    pixel_cols.reserve(row.size());

    auto is_all_digits = [](const std::string_view s) {
        if (s.empty()) return false;
        for (unsigned char ch : s) {
            if (!std::isdigit(ch)) return false;
        }
        return true;
    };

    for (std::size_t k = 0; k < row.size(); ++k) {
        std::string_view name = row[k];
        if (name == "label") {
            label_idx = k;
            continue;
        }
        if (name == "id") {
            id_idx = k;
            continue;
        }

        // match pattern "<digits>x<digits>" (e.g., 1x1, 28x28)
        auto pos = name.find('x');
        if (pos == std::string_view::npos) continue;

        std::string_view left = name.substr(0, pos);
        std::string_view right = name.substr(pos + 1);
        if (!is_all_digits(left) || !is_all_digits(right)) continue;

        int r = std::stoi(std::string(left));
        int c = std::stoi(std::string(right));
        pixel_cols.push_back(PixelCol{r, c, k});
    }

    if (has_label && label_idx == static_cast<std::size_t>(-1)) {
        std::cerr << "Could not find 'label' column in header\n";
        return false;
    }
    if (!has_label && id_idx == static_cast<std::size_t>(-1)) {
        std::cerr << "Could not find 'id' column in header\n";
        return false;
    }
    if (pixel_cols.empty()) {
        std::cerr << "No pixel columns detected in header\n";
        return false;
    }

    std::sort(pixel_cols.begin(), pixel_cols.end(), [](const PixelCol& a, const PixelCol& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    return true;
}
    

static bool load_mnist_data(
    std::ifstream& file,
    CSVRow& row,
    std::size_t label_idx,
    const std::vector<PixelCol>& pixel_cols,
    MnistData& data
) {
    const std::size_t n_features = pixel_cols.size();
    if (n_features == 0) {
        std::cerr << "No pixel columns detected in header\n";
        return false;
    }

    // --- Count rows (already consumed header) ---
    std::size_t n = 0;
    while (file >> row) ++n;

    // --- Allocate Armadillo matrices ---
    arma::mat X(n, n_features);
    arma::uvec y(n);

    // --- Rewind and read again ---
    file.clear();
    file.seekg(0);

    // skip header again
    file >> row;

    std::size_t i = 0;
    while (file >> row) {
        // label
        y(i) = static_cast<arma::uword>(std::stoul(std::string(row[label_idx])));

        // pixels/features
        for (std::size_t j = 0; j < n_features; ++j) {
            X(i, j) = std::stod(std::string(row[pixel_cols[j].idx]));
        }
        ++i;
    }

    // Normalization (pixels 0..255 -> 0..1)
    X /= 255.0;

    data.X = std::move(X);
    data.y = std::move(y);
    data.n = n;
    data.n_features = n_features;
    return true;
}

static bool load_mnist_test_data(
    std::ifstream& file,
    CSVRow& row,
    std::size_t id_idx,
    const std::vector<PixelCol>& pixel_cols,
    MnistTestData& data
) {
    const std::size_t n_features = pixel_cols.size();
    if (n_features == 0) {
        std::cerr << "No pixel columns detected in header\n";
        return false;
    }

    // --- Count rows (already consumed header) ---
    std::size_t n = 0;
    while (file >> row) ++n;

    // --- Allocate Armadillo matrix + ids ---
    arma::mat X(n, n_features);
    std::vector<std::string> ids(n);

    // --- Rewind and read again ---
    file.clear();
    file.seekg(0);

    // skip header again
    file >> row;

    std::size_t i = 0;
    while (file >> row) {
        ids[i] = std::string(row[id_idx]);

        // pixels/features
        for (std::size_t j = 0; j < n_features; ++j) {
            X(i, j) = std::stod(std::string(row[pixel_cols[j].idx]));
        }
        ++i;
    }

    // Normalization (pixels 0..255 -> 0..1)
    X /= 255.0;

    data.X = std::move(X);
    data.ids = std::move(ids);
    data.n = n;
    data.n_features = n_features;
    return true;
}

static void print_sanity_check(const MnistData& data) {
    // Small sanity-check
    std::cout << "Loaded rows: " << data.n << " | features: " << data.n_features << "\n";
    if (data.n > 0) {
        std::cout << "First label: " << data.y(0) << " | first pixel (after norm): " << data.X(0, 0) << "\n";
        std::cout << "X min/max: " << data.X.min() << " / " << data.X.max() << "\n";
    }
}

static void print_sanity_check(const MnistTestData& data) {
    // Small sanity-check
    std::cout << "Loaded rows: " << data.n << " | features: " << data.n_features << "\n";
    if (data.n > 0) {
        std::cout << "First id: " << data.ids[0] << " | first pixel (after norm): " << data.X(0, 0) << "\n";
        std::cout << "X min/max: " << data.X.min() << " / " << data.X.max() << "\n";
    }
}

static bool print_ascii_digit(const MnistData& data, std::size_t idx) {
    if (idx >= data.n) {
        std::cerr << "idx out of range\n";
        return false;
    }
    std::cout << "Label for idx " << idx << " = " << data.y(idx) << "\n";

    // Build a 28x28 image from the flattened row.
    arma::mat img_raw = arma::reshape(data.X.row(idx).t(), 28, 28);

    // Fix orientation: rotate 90° to the right and then mirror.
    // This is equivalent to a transpose of the 2D image.
    arma::mat img = img_raw.t();

    std::cout << "img(0,0)=" << img(0,0) << " img(27,27)=" << img(27,27) << "\n";

    auto shade = [](double v) {
        static const char* levels = " .:-=+*#%@";
        int idx = (int)std::round(v * 9.0);
        if (idx < 0) idx = 0;
        if (idx > 9) idx = 9;
        return levels[idx];
    };

    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c){
            std::cout << shade(img(r, c)) << shade(img(r, c));
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
    return true;
}

static bool train_nn(const MnistData& data, const double dropout_percent, NNModel& model){
    if (data.n == 0 || data.n_features == 0) {
        std::cerr << "Cannot train on empty dataset\n";
        return false;
    }

    if (dropout_percent < 0.0 || dropout_percent >= 1.0) {
        std::cerr << "dropout_percent must be in [0, 1)\n";
        return false;
    }

    arma::arma_rng::set_seed(1);
    const int iters = 5000;
    const double alpha = 0.05;
    const arma::uword hidden = 128;
    const arma::uword n_classes = 10;
    const arma::uword n_samples = data.X.n_rows;
    const double keep_prob = 1.0 - dropout_percent;

    model.best_acc = -1.0;
    model.W0.reset();
    model.W1.reset();
    model.b0.reset();
    model.b1.reset();

    // Weights: 784 inputs -> 1 output
    arma::mat syn0 = 2.0 * arma::randu<arma::mat>(data.n_features, hidden) - 1.0;
    arma::mat syn1 = 2.0 * arma::randu<arma::mat>(hidden, n_classes) - 1.0;

    arma::rowvec b0(hidden, arma::fill::zeros);
    arma::rowvec b1(n_classes, arma::fill::zeros);

    arma::mat Y(n_samples, n_classes, arma::fill::zeros);
    for (arma::uword i = 0; i < n_samples; ++i) {
        Y.row(i) = one_hot(static_cast<std::size_t>(data.y(i)), n_classes);
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < iters; ++i) {
        arma::mat l0 = data.X;

        arma::mat z1 = l0 * syn0;
        z1.each_row() += b0;
        arma::mat a1 = sigmoid(z1);
        arma::mat l1 = a1;

        arma::mat m1;
        if (keep_prob < 1.0) {
            m1 = dropout_mask(l1.n_rows, l1.n_cols, keep_prob);
            l1 %= m1;
        }

        arma::mat logits = l1 * syn1;
        logits.each_row() += b1;
        arma::mat p = softmax(logits);

        arma::mat l2_error = p - Y;

        arma::mat delta_syn1 = (l1.t() * l2_error) / static_cast<double>(n_samples);
        arma::rowvec delta_b1 = arma::mean(l2_error, 0);

        arma::mat hidden_grad = l2_error * syn1.t();
        if (keep_prob < 1.0) {
            hidden_grad %= m1;
        }
        hidden_grad %= sigmoid_deriv(a1);

        arma::mat delta_syn0 = (l0.t() * hidden_grad) / static_cast<double>(n_samples);
        arma::rowvec delta_b0 = arma::mean(hidden_grad, 0);

        syn1 -= alpha * delta_syn1;
        b1   -= alpha * delta_b1;
        syn0 -= alpha * delta_syn0;
        b0   -= alpha * delta_b0;

        if (!syn0.is_finite() || !syn1.is_finite() || !b0.is_finite() || !b1.is_finite()) {
            std::cerr << "Training diverged (non-finite weights) at iter " << i << "\n";
            return false;
        }

        arma::uvec preds = arma::index_max(p, 1);
        double acc = static_cast<double>(arma::accu(preds == data.y)) / static_cast<double>(n_samples);
        if (i % 200 == 0 || i == iters - 1) {
            arma::mat p_safe = arma::clamp(p, 1e-12, 1.0 - 1e-12);
            double loss = -arma::accu(Y % arma::log(p_safe)) / static_cast<double>(n_samples);
            std::cout << "Iter " << i << " | loss: " << loss << " | acc: " << acc << "\n";
        }
        
        
        if(acc > model.best_acc){
            model.best_acc = acc;
            model.W0 = syn0;
            model.W1 = syn1;
            model.b0 = b0;
            model.b1 = b1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Training time: " << ms << " ms (" << (ms / 1000.0) << " s)\n";
    return true;
};


static arma::mat predict_proba(const arma::mat& X, const NNModel& model){
    arma::mat z1 = X * model.W0;
    z1.each_row() += model.b0;

    arma::mat a1 = sigmoid(z1);

    arma::mat z2 = a1 * model.W1;
    z2.each_row() += model.b1;

    arma::mat p = softmax(z2);
    return p;
};


static arma::uvec predict(const arma::mat& X, const NNModel& model){
    arma::mat p = predict_proba(X, model);

    arma::uvec y_pred(p.n_rows);

    // argmax ligne par ligne
    for(arma::uword i = 0; i < p.n_rows; i++){
        y_pred(i) = p.row(i).index_max();
    }
    return y_pred;
};


static bool write_submission_csv(const std::vector<std::string>& ids, const arma::uvec& y_pred, const std::string& path){
    if (ids.size() != y_pred.n_elem) {
        std::cerr << "ids and predictions size mismatch\n";
        return false;
    }

    const char sep = ',';
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Impossible d'ouvrir le fichier\n";
        return false;
    }

    f.imbue(std::locale::classic());
    f << std::fixed << std::setprecision(6);

    f << "id" << sep << "label\n";

    for (std::size_t i = 0; i < ids.size(); i++){
        f << ids[i] << sep << y_pred[i] << "\n";
    }
    return true;
}



int main()
{
    std::ifstream file = open_mnist_file();
    if (!file) { return 1; }

    CSVRow row;
    if (!read_header(file, row)) { return 1; }

    std::size_t label_idx = static_cast<std::size_t>(-1);
    std::size_t id_idx = static_cast<std::size_t>(-1);
    std::vector<PixelCol> pixel_cols;
    if (!parse_header_columns(row, label_idx, pixel_cols, id_idx, true)) { return 1; }

    MnistData data;
    if (!load_mnist_data(file, row, label_idx, pixel_cols, data)) { return 1; }

    print_sanity_check(data);

    std::size_t idx = 123;
    if (!print_ascii_digit(data, idx)) { return 1; }

    NNModel model;
    if (!train_nn(data, 0.2, model)) { return 1; }

    std::ifstream file_test = open_test_mnist_file();
    if (!file_test) { return 1; }

    if (!read_header(file_test, row)) { return 1; }
    if (!parse_header_columns(row, label_idx, pixel_cols, id_idx, false)) { return 1; }
    MnistTestData test_data;
    if (!load_mnist_test_data(file_test, row, id_idx, pixel_cols, test_data)) { return 1; }
    print_sanity_check(test_data);

    arma::uvec y_pred = predict(test_data.X, model);
    if (y_pred.n_elem != test_data.n) {
        std::cerr << "Prediction size mismatch\n";
        return 1;
    }
    if (!write_submission_csv(test_data.ids, y_pred, "data/submission_cpp.csv")) { return 1; }
    std::cout << "submission.csv written\n";

    return 0;

}
