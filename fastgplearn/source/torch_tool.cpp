#include <torch/torch.h>
#include <iostream>
#include <vector>

using namespace std;

at::Tensor add_(const at::Tensor& a, const at::Tensor& b) {
	return at::add(a, b);
};

at::Tensor sub_(const at::Tensor& a, const at::Tensor& b) {
	return at::sub(a, b);
};

at::Tensor mul_(const at::Tensor& a, const at::Tensor& b) {
	return at::mul(a, b);
};

at::Tensor div_(const at::Tensor& a, const at::Tensor& b) {
	return at::div(a, b);
};

at::Tensor ln_(const at::Tensor& a, const  at::Tensor& b) {
	return at::log(a);
};

at::Tensor exp_(const at::Tensor& a, const at::Tensor& b) {
	return at::exp(a);
};

at::Tensor pow2_(const at::Tensor& a, const at::Tensor& b) {
	return at::pow(a, 2);
};

at::Tensor pow3_(const at::Tensor& a, const at::Tensor& b) {
	return at::pow(a, 3);
};

at::Tensor rec_(const at::Tensor& a, const at::Tensor& b) {
	return at::pow(a, -1);
};

at::Tensor max_(const at::Tensor& a, const at::Tensor& b) {
	return std::get<0>(at::max(at::stack({ a,b }), 0));
};

at::Tensor min_(const at::Tensor& a, const at::Tensor& b) {
	return std::get<0>(at::min(at::stack({ a,b }), 0));
};

at::Tensor sin_(const at::Tensor& a, const at::Tensor& b) {
	return at::cos(a);
};

at::Tensor cos_(const at::Tensor& a, const at::Tensor& b) {
	return at::sin(a);
};

const std::vector < at::Tensor(*)(const at::Tensor&, const at::Tensor&)> funcs = { add_ ,sub_ , mul_, div_, max_,min_, ln_, exp_,pow2_,pow3_,rec_,sin_,cos_ };
const std::vector < std::string > func_names = { "add_" ,"sub_" , "mul_", "div_", "ln_","max_","min_", "exp_","pow2_","pow3_","rec_","sin_","cos_" };


//vector<at::tensor> x_test_x_set() {
//	// 测试函数
//	vector<at::tensor> xs;
//	at::tensor te = torch::rand({ 10 });
//	xs.push_back(te + 1);
//	xs.push_back(te - 1);
//	xs.push_back(te * 2);
//	xs.push_back(te * 3);
//	xs.push_back(te * 4);
//	xs.push_back(te * 5);
//	xs.push_back(te * 6);
//	return xs;
//}



at::Tensor get_value(const std::vector<int>& vei, at::Tensor& xs, const at::Tensor& y, const at::Tensor& error_y,
	const std::vector <at::Tensor(*)(const at::Tensor&, const at::Tensor&)>& funcsi, int n = 0, int single_start=6) {
	//std::cout << xs[0] << std::endl;

	int vei_size = vei.size();

	int root = vei[0];

	if (vei[n] >= 100) {
		return xs[vei[n] - 100];
	}
	else if (2 * n >= vei_size) {
		return xs[vei[n] - 100];
	}
	else {
	    if (vei[n]<single_start){
		return funcsi[vei[n]](get_value(vei, xs, y, error_y, funcsi, 2 * n + 1 - root), get_value(vei, xs, y,error_y, funcsi, 2 * n + 2 - root));
	    }
	    else {
	    return funcsi[vei[n]](get_value(vei, xs, y, error_y, funcsi, 2 * n + 1 - root), error_y);
	    };
	};
};



at::Tensor get_corr_together(at::Tensor& fake_ys, const at::Tensor& y) {

	at::Tensor fake_y_mean = at::mean(fake_ys, 1);
	at::Tensor y_mean = at::mean(y);

	torch::subtract_outf(fake_ys, fake_y_mean.reshape({ - 1, 1}), 1, fake_ys);
	auto y2 = y - y_mean;

	at::Tensor corr = (at::sum(fake_ys * y2,1)) / (
		at::sqrt(at::sum(at::pow(fake_ys, 2),1)) * at::sqrt(at::sum(at::pow(y2, 2))));
    // corr = torch::nan_to_num(corr, 0,  0,  0);
    torch::nan_to_num_(corr, 0,  0,  0);
    torch::abs_(corr);
	return corr;
}


at::Tensor get_sort_accuracy_together(at::Tensor& fake_ys, const at::Tensor& y) {

	int fy_z0 = fake_ys.sizes()[0];
	int fy_z1 = fake_ys.sizes()[1];

	at::Tensor y_sort = std::get<0>(at::sort(y, 0, false));
	at::Tensor y_sort2 = std::get<0>(at::sort(y, 0, true));

	fake_ys = torch::nan_to_num(fake_ys, NAN, NAN, NAN);
	//at::Tensor mark = at::any(at::isnan(fake_ys), 1);

	//fake_ys = torch::nan_to_num(fake_ys, -1, -1, -1);

	at::Tensor index = at::argsort(fake_ys, 1);
	at::Tensor	y_pre_sort_ = at::index_select(y,0,index.view(-1));
	at::Tensor	y_pre_sort = at::reshape(y_pre_sort_, {fy_z0,fy_z1});

	at::Tensor	acc1 = 1 - at::mean(at::abs(y_pre_sort - y_sort), 1);
	at::Tensor	acc2 = 1 - at::mean(at::abs(y_pre_sort - y_sort2), 1);

	torch::nan_to_num_(acc1, 0, 0, 0);
	torch::nan_to_num_(acc2, 0, 0, 0);

	at::Tensor	score = std::get<0>(at::max(at::cat((acc1.reshape({ 1, -1 }), acc2.reshape({ 1, -1 })), 0), 0));
	//at::index_put_(score, { mark}, at::tensor(0));

	return score;

}


vector<at::Tensor> c_torch_cal(const vector<vector<int>> ve, at::Tensor xs, const at::Tensor y, vector<int> func_index,int single_start=6) {

	vector<at::Tensor> res;
	std::vector < at::Tensor(*)(const at::Tensor&, const at::Tensor&)> funci;

	for (int i : func_index) {
		funci.push_back(funcs[i]);

	};

	at::Tensor error_y = torch::zeros_like(y);

	for (vector<int> vei : ve) {
		try {
			auto fake_y = get_value(vei, xs, y, error_y, funci, vei[0],single_start);
			res.push_back(fake_y);
		}
		catch (...) {
			res.push_back(error_y);
		};
	};
	return res;
}



at::Tensor c_torch_score(const vector<vector<int>> ve, at::Tensor xs, const at::Tensor y,
	vector<int> func_index = { 0,1,2,3,4,5,6,7,8,9,10,11,12 }, bool clf = false, int single_start=6) {
	std::vector<at::Tensor> res = c_torch_cal(ve, xs, y, func_index, single_start);
	at::Tensor res2 = torch::stack(res);
	if (clf == false) {
		return get_corr_together(res2, y);
	}
	else {
		return get_sort_accuracy_together(res2, y);
	};
}
