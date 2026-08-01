在 H20 上对 Triton matmul kernel 做性能测试后，我的结论是：相比零散的代码级微优化，合理配置 autotune 更关键，尤其是 BLOCK_SIZE、num_warps 和 num_stages 这些核心参数组合，对吞吐和最终最优解影响最大。
