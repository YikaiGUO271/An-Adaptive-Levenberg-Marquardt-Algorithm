class OptimizerTester:
    """优化器测试框架"""
    def __init__(self, optimizers=None, dim=200, n_runs=1, max_iter=300, tol=1e-8,
                 problem_type="rosenbrock", q_param=3, p_param=2, lse_n=500,
                 lse_rho=0.5, tmle_n_samples=1000, tmle_df_true=5.0):
        self.optimizers = optimizers or [
            ARC(),
            AdaN(),
            Algorithm1(alpha=1/2,beta=1/6),
            Algorithm1Acc(),
            SuperUniversalNewton(),
            CR(),
            CubicMM(),
            ConvCubicMM(),
            ConvexCubicMM(),
        ]
        self.dim = dim
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.tol = tol
        self.problem_type = problem_type
        self.q_param = q_param
        self.p_param = p_param
        self.lse_n = lse_n
        self.lse_rho = lse_rho
        self.tmle_n_samples = tmle_n_samples
        self.tmle_df_true = tmle_df_true
        self.results = {}
        self.problem_names = {
            "rosenbrock": "Rosenbrock Function",
            "polytope": "Polytope Feasibility",
            "worst_instances": "Worst Instances Function",
            "zakharov": "Zakharov Function",
            "powell": "Powell Singular Function",
            "logsumexp": "Log-Sum-Exp Function",
            "tmle": "Multivariate t-MLE",  # 添加这一行
        }

    def _create_model(self):
        """根据问题类型创建模型"""
        if self.problem_type == "polytope":
            return PolytopeFeasibility(dim=self.dim, m=500, p=self.p_param)
        elif self.problem_type == "worst_instances":
            return WorstInstancesFunction(dim=self.dim, q=self.q_param)
        elif self.problem_type == "zakharov":
            return ZakharovFunction(dim=self.dim)
        elif self.problem_type == "powell":
            return PowellSingularFunction(dim=self.dim)
        elif self.problem_type == "logsumexp":
            return LogSumExpFunction(n=self.lse_n, d=self.dim, rho=self.lse_rho)
        elif self.problem_type == "tmle":  # 添加 t-MLE 问题
            return MultivariateTMLE(
                n_samples=self.tmle_n_samples,
                dim=self.dim,
                df_true=self.tmle_df_true,
                random_state=42
            )
        else:  # 默认Rosenbrock
            return HighDimRosenbrock(dim=self.dim)

    def _get_initial_theta(self, problem_type, run_idx):
        """根据问题类型获取初始点"""
        if problem_type == "polytope":
            # 多面体可行性问题，从原点开始
            return jnp.array(np.ones(self.dim) * 1)
        elif problem_type == "worst_instances":
            # Worst instances
            return jnp.array(np.ones(self.dim) * 1)
        elif problem_type == "zakharov":
            # Zakharov函数在原点有最小值，使用远离原点的初始点
            return jnp.array(np.ones(self.dim) * 1)
        elif problem_type == "powell":
            # Powell函数使用中等规模的初始点
            return jnp.array(np.ones(self.dim) * 1)
        elif problem_type == "logsumexp":
            # Powell函数使用中等规模的初始点
            return jnp.array(np.ones(self.dim) * 1)
        elif problem_type == "rosenbrock":  # Rosenbrock
            # Rosenbrock使用传统初始点
            return jnp.array(np.ones(self.dim) * 1.5)
        elif problem_type == "tmle":  # 添加 t-MLE 初始点
            # 使用 MultivariateTMLE 提供的初始猜测
            model = MultivariateTMLE(
                n_samples=self.tmle_n_samples,
                dim=self.dim,
                df_true=self.tmle_df_true,
                random_state=42
            )
            return model.get_initial_guess()

    def run_test(self, initial_theta_dict=None):
        """运行测试"""
        model = self._create_model()
        problem_name = self.problem_names.get(self.problem_type, "Rosenbrock Function")
        print(f"\nTesting on {problem_name} (dim={self.dim})")
        
        if initial_theta_dict is None:
            initial_theta_dict = {}
        
        for optimizer in self.optimizers:
            print(f"\nTesting {optimizer.name}...")
            self.results[optimizer.name] = []
            
            for run in range(self.n_runs):
                print(f"  Run {run + 1}/{self.n_runs}")
                
                # 获取初始点（优先使用自定义的）
                if self.problem_type in initial_theta_dict:
                    initial_theta = jnp.array(initial_theta_dict[self.problem_type])
                else:
                    initial_theta = self._get_initial_theta(self.problem_type, run)
                
                theta = optimizer.optimize(
                    model, self.dim,
                    initial_theta=initial_theta,
                    max_iter=self.max_iter
                )
                self.results[optimizer.name].append(optimizer.history.copy())
    
    def plot_results(self, time_range=None):
        """绘制结果 - 显示对数间隙距离
        Args:
            time_range: 时间范围 [min_time, max_time]，例如 [0, 10] 表示0-10秒
        """
        problem_name = self.problem_names.get(self.problem_type, "Rosenbrock Function")
        title = ""#f"{problem_name} (dim={self.dim}"
        #if self.problem_type == "worst_instances":
        #    title += f", q={self.q_param}"
        #elif self.problem_type == "polytope":
        #    title += f", p={self.p_param}"
        #title += ")"
        # 设置字体大小
        plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'axes.titlesize': 20,
        })
        # 获取问题的最优值（如果已知）
        optimal_values = {
            "rosenbrock": 0.0,           # Rosenbrock函数的最优值为0
            "zakharov": 0.0,             # Zakharov函数的最优值为0
            "worst_instances": 0.0,      # Worst instances函数的最优值为0
            "powell": 0.0,               # Powell函数的最优值为0
            "polytope": None,
            "logsumexp":None,
        }
        
        optimal_value = optimal_values.get(self.problem_type, 0.0)
        
        plt.figure(figsize=(16, 6))
        plt.suptitle(title, fontsize=14)
        
        # 定义线型样式
        styles = {
            'ARC': {'color': 'purple', 'linestyle': '--', 'linewidth': 2},
            'ALM': {'color': 'red', 'linestyle': '-', 'linewidth': 3},
            'SUN': {'color': 'green', 'linestyle': '--', 'linewidth': 2},
            'CR': {'color': 'black', 'linestyle': '--', 'linewidth': 2},
            'AdaN': {'color': 'orange', 'linestyle': '--', 'linewidth': 2},
            'CMM': {'color': 'brown', 'linestyle': '--',  'linewidth': 2},
            'ConvexCubicMM': {'color': 'black', 'linestyle': ':','linewidth': 6},
        }
    
        # 分别存储两个子图的收敛数据
        convergence_data_iter = {}  # 迭代次数的收敛数据
        convergence_data_time = {}  # 时间的收敛数据
        
        # 收集第一个子图的图例信息
        legend_handles_iter = []
        legend_labels_iter = []
        
        # 1. 对数间隙距离 vs 迭代
        plt.subplot(1, 2, 1)
        for name, runs in self.results.items():
            if not runs:
                continue
            
            # 找到最大迭代次数
            max_iterations = max(len(run['loss']) for run in runs if run['loss'])
            if max_iterations == 0:
                continue
                
            # 创建完整的间隙距离序列
            all_gaps = []
            all_times = []
            
            for run in runs:
                if run['loss'] and run['time']:
                    # 计算间隙距离：f(x) - f*
                    if optimal_value is not None:
                        gaps = [loss - optimal_value for loss in run['loss']]
                    else:
                        # 如果最优值未知，使用运行中的最小值作为参考
                        min_loss = min(run['loss'])
                        gaps = [loss - min_loss for loss in run['loss']]
                    
                    # 如果运行提前结束，用最后一个值填充
                    if len(gaps) < max_iterations:
                        padded_gaps = gaps + [gaps[-1]] * (max_iterations - len(gaps))
                        padded_times = run['time'] + [run['time'][-1]] * (max_iterations - len(run['time']))
                    else:
                        padded_gaps = gaps[:max_iterations]
                        padded_times = run['time'][:max_iterations]
                    
                    all_gaps.append(padded_gaps)
                    all_times.append(padded_times)
            
            if all_gaps:
                avg_gaps = np.mean(all_gaps, axis=0)
                avg_times = np.mean(all_times, axis=0)
                # 避免对数计算中的0或负值
                avg_gaps = np.maximum(avg_gaps, 1e-16)
                iterations = range(0,len(avg_gaps))
                line = plt.semilogy(iterations, avg_gaps, label=name, **styles.get(name, {}))[0]
                legend_handles_iter.append(line)
                legend_labels_iter.append(name)
                
                # 计算达到1e-12的迭代次数和时间
                convergence_iter = None
                convergence_time = None
                final_gap = avg_gaps[-1]  # 最终gap值
                
                for i, gap in enumerate(avg_gaps):
                    if gap <= 1e-12:
                        convergence_iter = i
                        if i < len(avg_times):
                            convergence_time = avg_times[i]
                        break
                
                # 如果没有达到收敛标准，使用最后一次迭代的数据
                if convergence_iter is None:
                    convergence_iter = len(avg_gaps) - 1
                    convergence_time = avg_times[-1]
                
                convergence_data_iter[name] = {
                    'iteration': convergence_iter,
                    'time': convergence_time,
                    'final_gap': final_gap,
                    'converged': convergence_iter < len(avg_gaps) - 1 or final_gap <= 1e-10
                }
        
        plt.xlabel('Iteration')
        plt.ylabel('Log Gap Distance: f(x) - f*')
        plt.grid(True, ls='-')
        plt.xlim(0, max_iterations*2)
        plt.ylim(bottom=1e-8)
        plt.title('Convergence vs Iterations')
        # 设置图例为两列
        plt.legend(handles=legend_handles_iter, labels=legend_labels_iter, ncol=2, 
                   loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 2. 对数间隙距离 vs 时间（不显示y轴刻度，但保留网格线）
        plt.subplot(1, 2, 2)
        for name, runs in self.results.items():
            if not runs:
                continue
            
            # 收集所有有效的时间-间隙数据
            valid_traces = []
            
            for run in runs:
                if run['loss'] and run['time']:
                    # 计算间隙距离：f(x) - f*
                    if optimal_value is not None:
                        gaps = [loss - optimal_value for loss in run['loss']]
                    else:
                        # 如果最优值未知，使用运行中的最小值作为参考
                        min_loss = min(run['loss'])
                        gaps = [loss - min_loss for loss in run['loss']]
                    
                    # 确保时间和gap长度一致
                    min_len = min(len(run['time']), len(gaps))
                    if min_len > 1:  # 至少需要2个点才能插值
                        times = run['time'][:min_len]
                        gaps_trunc = gaps[:min_len]
                        
                        # 确保时间是递增的
                        if times[-1] > times[0]:
                            valid_traces.append((times, gaps_trunc))
            
            if valid_traces:
                # 找到所有轨迹中的最大时间
                all_end_times = [trace[0][-1] for trace in valid_traces]
                if not all_end_times:  # 防止空列表
                    continue
                    
                max_time = max(all_end_times)
                
                # 确保时间范围有效
                if max_time <= 0:
                    continue
                    
                # 定义统一的时间采样点
                if time_range is None:
                    time_end = max_time
                else:
                    time_end = min(max_time, time_range[1])
                    
                # 确保采样点数合理
                num_samples = max(50, min(200, int(time_end * 10)))
                time_samples = np.linspace(0, time_end, num_samples)
                
                # 对每条轨迹进行插值
                interpolated_gaps_list = []
                
                for times_array, gaps_array in valid_traces:
                    # 确保数组是numpy数组
                    times_array = np.array(times_array)
                    gaps_array = np.array(gaps_array)
                    
                    # 处理间隙值，避免零或负值
                    gaps_array = np.maximum(gaps_array, 1e-16)
                    
                    if len(times_array) > 1:
                        # 在对数尺度上进行线性插值
                        log_gaps = np.log(gaps_array)
                        
                        # 只插值到该轨迹的时间范围内
                        trace_end_time = times_array[-1]
                        valid_samples = time_samples[time_samples <= trace_end_time]
                        
                        if len(valid_samples) > 0:
                            # 线性插值
                            interp_log = np.interp(valid_samples, times_array, log_gaps)
                            interp_gaps = np.exp(interp_log)
                            
                            # 如果轨迹提前结束，用最后一个值填充
                            if len(valid_samples) < len(time_samples):
                                fill_length = len(time_samples) - len(valid_samples)
                                interp_gaps = np.concatenate([
                                    interp_gaps, 
                                    np.full(fill_length, gaps_array[-1])
                                ])
                            else:
                                interp_gaps = interp_gaps[:len(time_samples)]
                                
                            interpolated_gaps_list.append(interp_gaps)
                
                if interpolated_gaps_list:
                    # 转换为数组并计算平均
                    interpolated_gaps_array = np.array(interpolated_gaps_list)
                    avg_gaps = np.mean(interpolated_gaps_array, axis=0)
                    
                    # 确保avg_gaps不为空
                    if len(avg_gaps) == 0:
                        continue
                        
                    # 避免零值
                    avg_gaps = np.maximum(avg_gaps, 1e-16)
                    
                    # 绘制曲线，不设置label（避免自动生成图例）
                    plt.semilogy(time_samples[:len(avg_gaps)], avg_gaps, 
                                **styles.get(name, {}))
                    
                    # 计算收敛时间和最终间隙
                    final_gap = avg_gaps[-1]
                    convergence_time = None
                    
                    for i, gap in enumerate(avg_gaps):
                        if gap <= 1e-12 and i < len(time_samples):
                            convergence_time = time_samples[i]
                            break
                    
                    # 记录收敛数据
                    convergence_data_time[name] = {
                        'time': convergence_time if convergence_time is not None else time_samples[-1],
                        'final_gap': final_gap,
                        'converged': convergence_time is not None
                    }
        
        plt.xlabel('Time (s)')
        plt.ylabel('')  # 清空y轴标签
        plt.grid(True, ls='-')
        
        # 隐藏y轴刻度，但保留网格线
        ax = plt.gca()
        ax.tick_params(axis='y', labelleft=False)  # 隐藏y轴刻度标签
        # 如果需要完全隐藏刻度线（包括小刻度），可以使用：
        # ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        # 但保留网格线不受影响
        
        if time_range:
            plt.xlim(0, time_range[1])
        plt.ylim(bottom=1e-8)
        plt.title('Convergence vs Time')
        # 不显示图例
        
        plt.tight_layout()
        plt.show()
        
        # 打印收敛性能表格
        self._print_convergence_tables(convergence_data_iter, convergence_data_time)
    
    def _print_convergence_tables(self, convergence_data_iter, convergence_data_time):
        """打印收敛性能表格"""
        print("\n" + "="*60)
        print("CONVERGENCE PERFORMANCE SUMMARY")
        print("="*60)
        
        # 迭代次数表格
        print("\n1. Convergence vs Iterations (Gap ≤ 1e-12)")
        print("-"*60)
        print(f"{'Algorithm':<20} {'Iterations':<12} {'Time (s)':<12} {'Final Gap':<15}")
        print("-"*60)
        
        for name, data in convergence_data_iter.items():
            iter_str = f"{data['iteration']}"
            time_str = f"{data['time']:.4f}"
            gap_str = f"{data['final_gap']:.2e}"
            
            print(f"{name:<20} {iter_str:<12} {time_str:<12} {gap_str:<15}")
        
        # 时间表格
        print("\n2. Convergence vs Time (Gap ≤ 1e-12)")
        print("-"*60)
        print(f"{'Algorithm':<20} {'Time (s)':<12} {'Final Gap':<15}")
        print("-"*60)
        
        for name, data in convergence_data_time.items():
            time_str = f"{data['time']:.4f}"
            gap_str = f"{data['final_gap']:.2e}"
            converged_str = "✓" if data['converged'] else "✗"
            
            print(f"{name:<20} {time_str:<12} {gap_str:<15} {converged_str}")
        
        print("="*60)
        
        # 找出最优算法
        self._identify_best_algorithms(convergence_data_iter, convergence_data_time)
    
    def _identify_best_algorithms(self, convergence_data_iter, convergence_data_time):
        """识别最优算法"""
        # 按迭代次数
        converged_iter = {name: data for name, data in convergence_data_iter.items() 
                         if data['converged']}
        
        # 按时间
        converged_time = {name: data for name, data in convergence_data_time.items() 
                         if data['converged']}
        
        if converged_iter:
            best_iter = min(converged_iter.items(), key=lambda x: x[1]['iteration'])
            print(f"\nFastest by iterations: {best_iter[0]} ({best_iter[1]['iteration']} iterations)")
        
        if converged_time:
            best_time = min(converged_time.items(), key=lambda x: x[1]['time'])
            print(f"Fastest by time: {best_time[0]} ({best_time[1]['time']:.4f} seconds)")
        
        if not converged_iter and not converged_time:
            # 如果没有算法收敛，找出最接近收敛的
            closest_iter = min(convergence_data_iter.items(), key=lambda x: x[1]['final_gap'])
            closest_time = min(convergence_data_time.items(), key=lambda x: x[1]['final_gap'])
            
            print(f"\nClosest to convergence by gap:")
            print(f"  By iterations: {closest_iter[0]} (gap: {closest_iter[1]['final_gap']:.2e})")
            print(f"  By time: {closest_time[0]} (gap: {closest_time[1]['final_gap']:.2e})")


from scipy.optimize import brentq, minimize_scalar
from jax.scipy.special import gammaln, digamma
class ECME(BaseOptimizer):
    """ECME算法 for t-distribution MLE - 直接参数操作版本"""
    
    def __init__(self, 
                 nu_min=0.01,
                 nu_max=100.0,
                 nu_init=10.0,
                 max_nu_iters=500,
                 nu_tol=1e-8):
        super().__init__('ECME')
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.nu_init = nu_init
        self.max_nu_iters = max_nu_iters
        self.nu_tol = nu_tol
    
    def _e_step(self, X, mu, Sigma, nu):
        """E-Step: 计算权重 w_i"""
        d = X.shape[1]
        diff = X - mu
        
        # 计算马氏距离
        try:
            L = jnp.linalg.cholesky(Sigma)
            solved = jax.vmap(lambda x: jax.scipy.linalg.solve_triangular(L, x, lower=True))(diff)
            mahal_sq = jnp.sum(solved**2, axis=1)
        except:
            Sigma_reg = Sigma + 1e-6 * jnp.eye(d)
            mahal_sq = jnp.sum(diff @ jnp.linalg.inv(Sigma_reg) * diff, axis=1)
        
        # 计算权重
        weights = (nu + d) / (nu + mahal_sq)
        return weights, mahal_sq
    
    def _cm_step_1(self, X, weights):
        """CM-Step 1: 更新 μ 和 Σ"""
        n = X.shape[0]
        d = X.shape[1]
        
        # 更新 μ
        weight_sum = jnp.sum(weights)
        mu_new = jnp.sum(weights[:, None] * X, axis=0) / weight_sum
        
        # 更新 Σ
        diff = X - mu_new
        Sigma_new = jnp.zeros((d, d))
        for i in range(n):
            Sigma_new += weights[i] * jnp.outer(diff[i], diff[i])
        Sigma_new = Sigma_new / n
        
        # 确保对称正定
        Sigma_new = (Sigma_new + Sigma_new.T) / 2
        min_eig = jnp.linalg.eigvalsh(Sigma_new)[0]
        if min_eig < 1e-8:
            Sigma_new += (1e-8 - min_eig + 1e-8) * jnp.eye(d)
        
        return mu_new, Sigma_new
    
    def _nu_equation(self, nu, d, n, weights, mahal_sq):
        """ν更新方程"""
        term1 = -digamma(nu/2) + jnp.log(nu/2) + 1
        term2 = jnp.mean(jnp.log(weights) - weights)
        term3 = digamma((nu + d)/2) - jnp.log((nu + d)/2)
        return term1 + term2 + term3
    
    def _cm_step_2(self, X, mu, Sigma, nu_current, d):
        """CM-Step 2: 更新 ν"""
        n = X.shape[0]
        weights, mahal_sq = self._e_step(X, mu, Sigma, nu_current)
        
        # 定义目标函数
        def nu_objective(nu):
            if nu <= self.nu_min or nu >= self.nu_max:
                return 1e10
            term1 = n * (gammaln((nu + d)/2) - gammaln(nu/2))
            term2 = -n * d/2 * jnp.log(nu * jnp.pi)
            term3 = jnp.sum(((nu + d)/2) * jnp.log(1 + mahal_sq/nu))
            return -(term1 + term2 - term3)
        
        # 使用Brent方法
        try:
            def f(nu):
                return self._nu_equation(nu, d, n, weights, mahal_sq)
            
            if f(self.nu_min) * f(self.nu_max) < 0:
                nu_new = brentq(f, self.nu_min, self.nu_max, xtol=self.nu_tol)
            else:
                result = minimize_scalar(nu_objective, 
                                       bounds=(self.nu_min, self.nu_max), 
                                       method='bounded')
                nu_new = result.x
        except:
            result = minimize_scalar(nu_objective, 
                                   bounds=(self.nu_min, self.nu_max), 
                                   method='bounded')
            nu_new = result.x
        
        return float(nu_new)
    
    def _compute_loss_direct(self, X, mu, Sigma, nu):
        """直接计算损失（不通过模型）"""
        n, d = X.shape
        
        # 常数项
        log_gamma_term = gammaln((nu + d)/2) - gammaln(nu/2)
        log_det_term = 0.5 * jnp.linalg.slogdet(Sigma)[1]
        log_const = log_gamma_term - 0.5 * d * jnp.log(nu * jnp.pi) - log_det_term
        
        # 马氏距离
        mahal_sq = self._e_step(X, mu, Sigma, nu)[1]
        
        # 数据项
        data_terms = (nu + d)/2 * jnp.log(1 + mahal_sq / nu)
        total_data_term = jnp.sum(data_terms)
        
        # 负对数似然
        nll = -n * log_const + total_data_term
        return float(nll)
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        """ECME优化 - 直接操作参数版本"""
        self.reset_history()
        
        # 获取数据
        X = model.X
        d = model.dim
        
        # 初始化参数
        if initial_theta is not None:
            mu, L_flat, nu_tilde = model._unpack_parameters(initial_theta)
            L = model._reconstruct_L(L_flat)
            Sigma = L @ L.T
            nu = jnp.exp(nu_tilde)
        else:
            mu = jnp.mean(X, axis=0)
            Sigma = jnp.cov(X, rowvar=False) + 1e-6 * jnp.eye(d)
            nu = self.nu_init
        
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"ECME Optimization for t-MLE")
        print(f"{'='*60}")
        print(f"{'Iter':>6} {'Loss':>15} {'nu':>10} {'mu_change':>12} {'Sigma_change':>12}")
        print(f"{'-'*67}")
        
        for iteration in range(max_iter):
            # 保存旧参数
            mu_old, Sigma_old, nu_old = mu, Sigma, nu
            
            # E-Step 和 CM-Steps
            weights, _ = self._e_step(X, mu, Sigma, nu)
            mu, Sigma = self._cm_step_1(X, weights)
            nu = self._cm_step_2(X, mu, Sigma, nu, d)
            
            # 计算参数变化
            mu_change = jnp.linalg.norm(mu - mu_old)
            Sigma_change = jnp.linalg.norm(Sigma - Sigma_old) / (d * d)
            
            # 计算损失（用于记录）
            loss = self._compute_loss_direct(X, mu, Sigma, nu)
            
            # 记录历史
            self.history['loss'].append(loss)
            self.history['grad_norm'].append(0.0)  # ECME不计算梯度
            self.history['time'].append(time.time() - start_time)
            
            # 打印进度
            if iteration % 10 == 0 or iteration < 5:
                print(f"{iteration:6d} {loss:15.6e} {nu:10.4f} {mu_change:12.4e} {Sigma_change:12.4e}")
            
            # 检查收敛
            if mu_change < 1e-16 and Sigma_change < 1e-16:
                print(f"\nConverged at iteration {iteration}: small parameter changes")
                break
        
        print(f"{'-'*67}")
        print(f"Final: loss={loss:.6e}, nu={nu:.4f}")
        
        # 打包返回参数
        return self._pack_parameters(model, mu, Sigma, nu)
    
    def _pack_parameters(self, model, mu, Sigma, nu):
        """打包参数（仅在最后使用）"""
        try:
            L = jnp.linalg.cholesky(Sigma)
        except:
            Sigma_reg = Sigma + 1e-6 * jnp.eye(model.dim)
            L = jnp.linalg.cholesky(Sigma_reg)
        
        L_flat = []
        for i in range(model.dim):
            for j in range(i + 1):
                if i == j:
                    L_flat.append(float(jnp.log(jnp.maximum(L[i, j], 1e-8))))
                else:
                    L_flat.append(float(L[i, j]))
        
        nu_tilde = float(jnp.log(jnp.maximum(nu, 1e-8)))
        return np.concatenate([np.array(mu), np.array(L_flat), np.array([nu_tilde])])


class MultivariateTTest:
    """多元t分布MLE专门测试类"""
    
    def __init__(self, optimizers=None, dim=10, n_samples=1000, df_true=5.0, max_iter=100):
        self.optimizers = optimizers or [
            CR(),
            ARC(),
            AdaN(),
            Algorithm1(alpha=1/2,beta=1/3),
            Algorithm1Acc(alpha=1/2,beta=1/3),
            SuperUniversalNewton(),
            CubicMM(L_fixed=1e4),
            ECME(),
        ]
        self.dim = dim
        self.n_samples = n_samples
        self.df_true = df_true
        self.max_iter = max_iter
        self.results = {}
        
        # 创建多元t分布MLE问题实例
        self.model = MultivariateTMLE(
            n_samples=n_samples, 
            dim=dim, 
            df_true=df_true
        )
        
        # 存储最优损失值（用于计算间隙）
        self.optimal_loss = None
    
    def run_test(self):
        """运行多元t分布MLE测试"""
        print(f"\nTesting Multivariate t-distribution MLE")
        print(f"Dimension: {self.dim}, Samples: {self.n_samples}, True df: {self.df_true}")
        print(f"Stopping criterion: gradient norm < 1e-8")
        
        initial_theta = self.model.get_initial_guess()
        param_count = self.model.get_parameter_count()
        print(f"Total parameters to estimate: {param_count}")
        
        for optimizer in self.optimizers:
            print(f"\nTesting {optimizer.name}...")
            
            # 运行优化
            theta_opt = optimizer.optimize(
                self.model, 
                param_count,
                initial_theta=initial_theta,
                max_iter=self.max_iter,
                tol=1e-7  # 修改停止准则为梯度范数 < 1e-8
            )
            
            # 评估估计质量
            estimation_quality = self.model.evaluate_estimation(theta_opt)
            
            self.results[optimizer.name] = {
                'history': optimizer.history.copy(),
                'estimation_quality': estimation_quality,
                'final_theta': theta_opt
            }
            
            print(f"  Final loss: {optimizer.history['loss'][-1]:.6f}")
            print(f"  Final gradient norm: {optimizer.history['grad_norm'][-1]:.6e}")
            print(f"  Mu error: {estimation_quality['mu_error']:.6f}")
            print(f"  Sigma error: {estimation_quality['Sigma_error']:.6f}")
            print(f"  Nu error: {estimation_quality['nu_error']:.6f}")
        
        # 计算最优损失值（所有算法中的最小值）
        all_losses = []
        for result in self.results.values():
            history = result['history']
            if history['loss']:
                all_losses.extend(history['loss'])
        self.optimal_loss = min(all_losses) if all_losses else 0
    
    def plot_results(self, iter_range=None, time_range=None, figsize=(16, 6), wspace=0.05):
        """
        绘制多元t分布MLE的优化结果 - 显示对数间隙距离
        
        参数:
        iter_range: tuple (start, end) 迭代次数显示范围，None表示自动，例如 (0, 50)
        time_range: tuple (start, end) 时间显示范围，None表示自动，例如 (0, 10)
        figsize: tuple (width, height) 图像大小，默认(16, 6)
        wspace: float 子图之间的间距，默认0.05（很小的间距）
        """
        problem_name = f"Multivariate t-distribution MLE"
        title = f"{problem_name} (dim={self.dim}, n_samples={self.n_samples}, df_true={self.df_true})"
        
        # 设置字体大小
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
        })
        
        # 定义线型样式
        styles = {
            'ARC': {'color': 'purple', 'linestyle': '--', 'linewidth': 3},
            'AdaN': {'color': 'orange', 'linestyle': '--', 'linewidth': 3},
            'ALM': {'color': 'red', 'linestyle': '-', 'linewidth': 3},
            'ALM-a': {'color': 'red', 'linestyle': '--', 'linewidth': 3},
            'Algorithm1Acc': {'color': 'red', 'linestyle': '--', 'linewidth': 3},
            'Algorithm2': {'color': 'blue', 'linestyle': '-', 'linewidth': 3},
            'Algorithm3': {'color': 'cyan', 'linestyle': '-', 'linewidth': 3},
            'SUN': {'color': 'green', 'linestyle': '--', 'linewidth': 3},
            'CR': {'color': 'black', 'linestyle': '--', 'linewidth': 3},
            'CMM': {'color': 'brown', 'linestyle': '--', 'linewidth': 3},
            'ECME': {'color': 'black', 'linestyle': ':', 'linewidth': 3},
        }
        
        # 创建图像，设置子图间距
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plt.subplots_adjust(wspace=wspace)  # 设置子图间距
        
        # 分别存储两个子图的收敛数据
        convergence_data_iter = {}  # 迭代次数的收敛数据
        convergence_data_time = {}  # 时间的收敛数据
        
        # 收集所有算法的对数间隙数据，用于确定Y轴范围
        all_gaps_iter = []
        all_gaps_time = []
        
        # 1. 对数间隙距离 vs 迭代
        for name, result in self.results.items():
            history = result['history']
            if history['loss'] and history['time']:
                losses = history['loss']
                times = history['time']
                
                # 计算间隙并取对数
                gaps = []
                for loss in losses:
                    gap = max(loss - self.optimal_loss, 1e-16)  # 避免负值和零值
                    gaps.append(np.log10(gap))
                
                # 确保长度一致
                min_len = min(len(gaps), len(times))
                if min_len > 0:
                    gaps = gaps[:min_len]
                    times = times[:min_len]
                    
                    # 存储所有间隙数据
                    all_gaps_iter.extend(gaps)
                    
                    # 绘制对数间隙曲线
                    iterations = range(len(gaps))
                    ax1.plot(iterations, gaps, label=name, **styles.get(name, {}))
                    
                    # 计算收敛数据（达到目标间隙）
                    target_gap = np.log10(1e-12)  # 目标间隙为1e-6，对应log10 = -6
                    convergence_iter = None
                    convergence_time = None
                    final_gap = gaps[-1] if gaps else 0
                    
                    for i, gap in enumerate(gaps):
                        if gap <= target_gap:
                            convergence_iter = i
                            if i < len(times):
                                convergence_time = times[i]
                            break
                    
                    # 如果没有达到收敛标准，使用最后一次迭代的数据
                    if convergence_iter is None:
                        convergence_iter = len(gaps) - 1
                        convergence_time = times[-1] if times else 0
                    
                    convergence_data_iter[name] = {
                        'iteration': convergence_iter,
                        'time': convergence_time,
                        'final_gap': final_gap,
                        'converged': convergence_iter < len(gaps) - 1 or final_gap <= target_gap
                    }
        
        ax1.set_xlabel('Iteration', fontsize=14)
        ax1.set_ylabel(r'$f(x) - f^*$ (log scale)', fontsize=14)
        ax1.grid(True, alpha=0.6)
        
        # 设置Y轴刻度标签为10^x形式
        def format_y_tick(value, pos):
            """将数值格式化为10^x形式"""
            return f'$10^{{{int(value)}}}$'
        
        # 设置Y轴范围：从起始点高度开始
        if all_gaps_iter:
            # 找到起始点的最大值（即第一个点）
            start_gaps = []
            for name, result in self.results.items():
                history = result['history']
                if history['loss']:
                    losses = history['loss']
                    if losses:
                        gap = max(losses[0] - self.optimal_loss, 1e-16)
                        start_gaps.append(np.log10(gap))
            
            if start_gaps:
                y_max = max(start_gaps) + 0.5  # 在最高起始点基础上加一点余量
                y_min = -12  # 下限
                ax1.set_ylim(bottom=y_min, top=y_max)
                
                # 设置Y轴刻度
                y_ticks = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1, 2)
                ax1.set_yticks(y_ticks)
                ax1.set_yticklabels([format_y_tick(tick, None) for tick in y_ticks])
        
        # 设置迭代次数显示范围
        if iter_range is not None:
            ax1.set_xlim(iter_range[0], iter_range[1])
        
        ax1.set_title('Log Gap vs Iterations', fontsize=15)
        # 设置图例为两列，放在左下角避免遮挡
        ax1.legend(ncol=2, loc='lower left', framealpha=0.9, fontsize=11)
        
        # 2. 对数间隙距离 vs 时间
        for name, result in self.results.items():
            history = result['history']
            if history['loss'] and history['time']:
                losses = history['loss']
                times = history['time']
                
                # 计算间隙并取对数
                gaps = []
                for loss in losses:
                    gap = max(loss - self.optimal_loss, 1e-16)
                    gaps.append(np.log10(gap))
                
                # 确保时间和损失长度一致
                min_len = min(len(times), len(gaps))
                if min_len > 1:  # 至少需要2个点才能插值
                    times = times[:min_len]
                    gaps = gaps[:min_len]
                    
                    # 存储所有间隙数据
                    all_gaps_time.extend(gaps)
                    
                    # 确保时间是递增的
                    if times[-1] > times[0]:
                        # 创建时间采样点
                        if time_range is None:
                            time_end = times[-1]
                        else:
                            time_end = min(times[-1], time_range[1])
                        
                        # 动态采样点数
                        num_samples = max(50, min(200, int(time_end * 10)))
                        time_samples = np.linspace(0, time_end, num_samples)
                        
                        # 插值
                        interp_gaps = np.interp(time_samples, times, gaps)
                        
                        # 绘制曲线
                        ax2.plot(time_samples, interp_gaps, 
                                **styles.get(name, {}))
                        
                        # 计算收敛数据
                        target_gap = np.log10(1e-12)
                        final_gap = interp_gaps[-1]
                        convergence_time = None
                        
                        for i, gap in enumerate(interp_gaps):
                            if gap <= target_gap and i < len(time_samples):
                                convergence_time = time_samples[i]
                                break
                        
                        convergence_data_time[name] = {
                            'time': convergence_time if convergence_time is not None else time_samples[-1],
                            'final_gap': final_gap,
                            'converged': convergence_time is not None
                        }
        
        ax2.set_xlabel('Time (seconds)', fontsize=14)
        # 去掉第二张图的y轴标签，但保留刻度线
        ax2.set_ylabel('')  # 清空y轴标签
        ax2.grid(True, ls='-', alpha=1)
        
        # 设置Y轴范围并保留刻度线，但隐藏刻度标签
        if all_gaps_time:
            # 找到起始点的最大值
            start_gaps_time = []
            for name, result in self.results.items():
                history = result['history']
                if history['loss'] and history['time']:
                    losses = history['loss']
                    if losses:
                        gap = max(losses[0] - self.optimal_loss, 1e-16)
                        start_gaps_time.append(np.log10(gap))
            
            if start_gaps_time:
                y_max = max(start_gaps_time) + 0.5
                y_min = -12
                ax2.set_ylim(bottom=y_min, top=y_max)
                
                # 设置Y轴刻度位置
                y_ticks = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1, 2)
                ax2.set_yticks(y_ticks)
                # 隐藏刻度标签，但保留刻度线
                ax2.set_yticklabels([])
                # 确保刻度线仍然显示
                ax2.tick_params(left=True, labelleft=False)
        
        # 设置时间显示范围
        if time_range is not None:
            ax2.set_xlim(time_range[0], time_range[1])
        
        ax2.set_title('Log Gap vs Time', fontsize=15)
        # 右子图不显示图例
        
        plt.tight_layout()
        plt.show()
        
        # 打印收敛性能表格
        self._print_convergence_tables(convergence_data_iter, convergence_data_time)
    
    def _print_convergence_tables(self, convergence_data_iter, convergence_data_time):
        """打印收敛性能表格"""
        print("\n" + "="*70)
        print("CONVERGENCE PERFORMANCE SUMMARY")
        print(f"Reference optimal loss: {self.optimal_loss:.6f}")
        print("Convergence criterion: log10 gap ≤ -6 (i.e., f(x) - f* ≤ 1e-6)")
        print("="*70)
        
        # 迭代次数表格
        print("\n1. Convergence vs Iterations:")
        print("-"*70)
        print(f"{'Algorithm':<25} {'Iterations':<12} {'Time (s)':<12} {'Final Log Gap':<18}")
        print("-"*70)
        
        for name in sorted(convergence_data_iter.keys()):
            data = convergence_data_iter[name]
            iter_str = f"{data['iteration']}"
            time_str = f"{data['time']:.4f}"
            gap_str = f"{data['final_gap']:.6f}"
            
            print(f"{name:<25} {iter_str:<12} {time_str:<12} {gap_str:<18}")
        
        # 时间表格
        print("\n2. Convergence vs Time:")
        print("-"*70)
        print(f"{'Algorithm':<25} {'Time (s)':<12} {'Final Log Gap':<18} {'Status':<10}")
        print("-"*70)
        
        for name in sorted(convergence_data_time.keys()):
            data = convergence_data_time[name]
            time_str = f"{data['time']:.4f}"
            gap_str = f"{data['final_gap']:.6f}"
            converged_str = "✓" if data['converged'] else "✗"
            
            print(f"{name:<25} {time_str:<12} {gap_str:<18} {converged_str:<10}")
        
        print("="*70)
        
        # 找出最优算法
        self._identify_best_algorithms(convergence_data_iter, convergence_data_time)
    
    def _identify_best_algorithms(self, convergence_data_iter, convergence_data_time):
        """识别最优算法"""
        # 按迭代次数
        converged_iter = {name: data for name, data in convergence_data_iter.items() 
                         if data['converged']}
        
        # 按时间
        converged_time = {name: data for name, data in convergence_data_time.items() 
                         if data['converged']}
        
        if converged_iter:
            best_iter = min(converged_iter.items(), key=lambda x: x[1]['iteration'])
            print(f"\nFastest by iterations: {best_iter[0]} ({best_iter[1]['iteration']} iterations)")
        
        if converged_time:
            best_time = min(converged_time.items(), key=lambda x: x[1]['time'])
            print(f"Fastest by time: {best_time[0]} ({best_time[1]['time']:.4f} seconds)")
        
        # 按最终间隙质量排序
        print("\nRanking by final log gap quality:")
        print("-"*50)
        sorted_by_gap = sorted(convergence_data_iter.items(), key=lambda x: x[1]['final_gap'])
        for i, (name, data) in enumerate(sorted_by_gap[:5], 1):  # 显示前5个
            print(f"{i}. {name:<22} final log gap: {data['final_gap']:.6f}")
        
        if not converged_iter and not converged_time:
            print("\nNo algorithm reached convergence criterion")
            # 找出最接近收敛的
            closest_iter = min(convergence_data_iter.items(), key=lambda x: x[1]['final_gap'])
            closest_time = min(convergence_data_time.items(), key=lambda x: x[1]['final_gap'])
            
            print(f"\nClosest to convergence by log gap:")
            print(f"  By iterations: {closest_iter[0]} (log gap: {closest_iter[1]['final_gap']:.6f})")
            print(f"  By time: {closest_time[0]} (log gap: {closest_time[1]['final_gap']:.6f})")
