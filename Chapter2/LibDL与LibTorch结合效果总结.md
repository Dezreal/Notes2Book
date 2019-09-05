# LibDL与LibTorch结合效果总结

### 方法思路

旧版框架的问题主要在于运行效率低下，当时将原因主要归于，在双向计算中，当算法中存在循环时，JNI的调用次数过多。所以本次尝试将`Operator`的运算过程移到c++代码层，通过一次性传参，调用native方法直接取出运算结果来解决这一问题。

#### 旧版的Unfold微分计算

```java
                new OperandInfo(input, () -> {

                    assert input.data.rank() == 4;

                    long[] shape = input.data.shape();

                    INDArray zeros = Nd4j.zeros(shape[0], shape[1], shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
                    INDArray result = zeros.dup();

                    INDArray column;
                    for (long i = 0; i < amount_h; i++) {
                        for (long j = 0; j < amount_w; j++) {
                            column = grad.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i*amount_w+j))
                                    .reshape(shape[0], shape[1], filter_h, filter_w);
                            zeros.put(new INDArrayIndex[] {
                                    NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
                                    NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])
                            }, column);
                            result.addi(zeros);
                            zeros.muli(0);
                        }
                    }

                    result = result.get(
                            NDArrayIndex.all(),
                            NDArrayIndex.all(),
                            NDArrayIndex.interval(padding[0], padding[0]+shape[2]),
                            NDArrayIndex.interval(padding[1], padding[1]+shape[3]));

                    return result;
                })
```

#### 将其移至c++

java部分

```java
                new OperandInfo(input, () -> {

                    assert input.data.rank() == 4;

                    LibDL.core.Tensor input = Trans.INDArray2Tensor(this.input.data); //临时的数据转换方法
                    LibDL.core.Tensor grad = Trans.INDArray2Tensor(this.grad).to(Dtype.FLOAT32); //临时的数据转换方法

                    LibDL.core.Tensor out = ld.unfold(input, grad, padding[0], padding[1], (int)amount_h, (int)amount_w,
                            filter_h, filter_w, stride[0], stride[1], dilation[0], dilation[1]);
                    
                    INDArray result = Trans.Tensor2INDArray(out); //临时的数据转换方法
                    return result;
                })
```

c++部分

```c++
    static Tensor unfold(
            const Tensor& input, const Tensor& grad, int padding0, int padding1, long amount_h, long amount_w,
            int filter_h, int filter_w, int stride0, int stride1, int dilation0, int dilation1)
            throw(std::exception) {

        INDArray ci = input.core;
        INDArray g = grad.core;

        const long *shape = ci.sizes().data();

        const int64_t z[4] = {*shape, *(shape + 1), *(shape + 2) + padding0 * 2, *(shape + 3) + padding1 * 2};

        INDArray zeros = torch::zeros(at::IntArrayRef(z, 4));
        INDArray result = zeros.clone();

        INDArray column;
        int l = z[0] * z[1] * filter_h * filter_w;

        for (long i = 0; i < amount_h; i++) {
            for (long j = 0; j < amount_w; j++) {
                column = g.narrow(2, i * amount_w + j, 1).reshape({z[0], z[1], filter_h, filter_w});
                INDArray i0 = torch::arange(0, shape[0]).to(at::kLong);
                INDArray i1 = torch::arange(0, shape[1]).to(at::kLong);
                INDArray i2 = torch::arange(i * stride0, i * stride0 + filter_h * dilation0, dilation0).to(at::kLong);
                INDArray i3 = torch::arange(j * stride1, j * stride1 + filter_w * dilation1, dilation1).to(at::kLong);

                INDArray* src = &zeros;
                put(src, column, i0, i1, i2, i3);
                result.add_(zeros);
                zeros.mul_(0);
            }
        }

        result = result
                .index_select(2, torch::arange(padding0, padding0 + shape[2]).to(at::kLong))
                .index_select(3, torch::arange(padding1, padding1 + shape[3]).to(at::kLong));

        return result;
    }
```

### 期望效果

从两种实现方式中可以看出，算法公式完全一致，而第二种方法只有一次JNI调用，远远小于第一种方法（N2次JNI调用），因此期望第二种方式运行更快。

### 实际效果

遗憾的是实际测试结果与预期完全相反，第二种方法的效率大幅落后于原先的方法。

### 原因分析

经测试，第二种方法的80%以上的时间用于`put`方法的索引处理。事实上，torch的`index_put_`方法与nd4j的`put`方法在index参数的格式上有所不同，因此中间需要调用一个索引转换的方法。

put与索引转换

```c++
    static INDArray ix(const at::Tensor& src, int w, int h) throw (std::exception) {
        INDArray src_ = src.reshape({-1, 1});
        INDArray vh[h];

        for (int i = 0; i < h; i++) {
            *(vh + i) = src_;
        }
        at::TensorList tensorList(vh, h);
        INDArray result = torch::cat(tensorList, 0);
        INDArray vw[w];
        for (int i = 0; i < w; i++) {
            *(vw + i) = result;
        }
        tensorList = at::TensorList(vw, w);
        result = torch::cat(tensorList, 1);
        return result.reshape(at::IntArrayRef({1, -1})).squeeze();
    };
    static void put(INDArray* src, const INDArray& value, INDArray i0, INDArray i1, INDArray i2, INDArray i3) {
        int l0 = i0.size(0);
        int l1 = i1.size(0);
        int l2 = i2.size(0);
        int l3 = i3.size(0);
        int l = l0*l1*l2*l3;
        i0 = ix(i0, l3*l2*l1, l/(l/l0)/l0);
        i1 = ix(i1, l3*l2, l/(l3*l2)/l1);
        i2 = ix(i2, l3, l/l3/l2);
        i3 = ix(i3, 1, l/1/l3);
        src->index_put_(at::TensorList(std::array<INDArray, 4>{i0, i1, i2, i3}), value.reshape({-1}));
    }
```

但也不能说解决了这个索引的问题，就万事大吉了，一方面，即使除去这80%的时间，第二种方法仍稍微落后于原先的方法；另一方面，Nd4j的`put`方法也存在索引处理的过程，且会占用一定的时间，这在几个月前我也提到过。所以说，索引上的时间不能说是完全不合理的（在算法不变的基础上）。

至于为什么更少的JNI调用反而占用了更多的时间，这个我仍然不解，一种可能的原因是基于Nd4j的API特点而实现的算法不是非常适合直接翻译成torch版本，例如，翻译后的代码需要多处调用`to`等方法来满足格式要求，以及二者差异较大的"index"系方法。

### 个人想法

我尝试这个方案的原因其实是对当前的项目方向有所担心，期望能将我们自己的框架与且只与torch的线性计算部分集合起来，而不是全盘依赖libtorch。

将几个Operator的计算公式下沉进行测试是第一步，第二步是对已有的Operator进行调研，对它们进行拆分或组合，以及“自动”、“手动”微分的选择，然后根据torch的API特点设计计算算法。

我的想法有两方面：一方面，现在第一步遇到了较大的问题，但还是有解决的希望，可以通过进行更多的对比实验来分析问题到底是不是在算法的设计上，也可以调研torch的operator，看能否调用他们的计算接口；另一方面，从总体上看，这个方案风险仍然较大，各种实验和调研将花费很多时间，短时间内不一定能有进展，而且后续如果要我们自己设计双向的计算算法，对项目组成员的能力也有很高的要求（据我的感受，算法中一两个接口用的不合适，就可能造成10%以上的性能损失）。

现在项目组人手比较缺乏，我觉得先沿着当前项目的计划进行下去比较合适。