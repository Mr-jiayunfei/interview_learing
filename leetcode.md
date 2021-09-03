# stl常用算法

## 求和

std::accumulate(num.begin(), num.end(), 0); // std::accumulate 可以很方便



# 基本概念

## B+树

## B-树

## 红黑树

## 二叉搜索树

1 根节点的值大于左子树所有节点的值，小于右子树所有节点的值，且左子树和右子树也同样为二叉搜索树。

## 二叉树

除叶子结点外，树的每个节点的子节点数量不大于2

## 满二叉树

二叉树除了叶结点外所有节点都有两个子节点。

## 完全二叉树

从根往下数，除了最下层外都是全满（都有两个子节点），而最下层所有叶结点都向左边靠拢填满。

![image-20210818142158786](https://i.loli.net/2021/08/18/hVe9DPzZ8YGuByN.png)



# 生产者消费者

```c++
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <mutex>
#include <condition_variable>

using namespace std;
```



## 一个生产者一个消费者

```c++
//一个生产者一个消费者
class QueueOneConsumerOneProducer
{
public:
	QueueOneConsumerOneProducer(){}	
	void Enqueue(int data)
	{
		//不需要考虑queue满的情况
		qi.push(data);
	}
	
	int Dequeue()
	{
		if(qi.empty())
		{
			return -1;
		}
	
		int ret = qi.front();
		qi.pop();
		return ret;
	}

private:
	queue<int> qi;
};
```

## 多个生产者一个消费者

```c++
//多个生产者，一个消费者

const int MAX_SIZE =100;

class QueueMoreConsumerOneProducer

{

public:

	QueueMoreConsumerOneProducer(){}
	
	void Enqueue(int data)
	{
		unique_lock<mutex> lock(m_mtx);
		//需要考虑queue满的情况
		while(qi.size() == MAX_SIZE)
		{
			m_cv_not_full.wait(lock);
		}
		qi.push(data);
	}
	
	int Dequeue()
	{
		if(qi.empty())
		{
			return -1;
		}
	
		int ret = qi.front();
		qi.pop();
		m_cv_not_full.notify_one();
		return ret;
	}

private:
	queue<int> qi;
	mutex m_mtx;
	condition_variable m_cv_not_full;
};
```

## 一个生产者多个消费者

```c++
//一个生产者，多个消费者
class QueueOneConsumerMoreProducer
{
public:
	QueueOneConsumerMoreProducer(){}
	
	void Enqueue(int data)
	{
		qi.push(data);
		m_cv_not_empty.notify_one();
	}
	
	int Dequeue()
	{
		unique_lock<mutex> lock(m_mtx);
		while(qi.size()==0)
		{
			m_cv_not_empty.wait(lock);
		}
	
		int ret = qi.front();
		qi.pop();
		return ret;
	}

private:
	queue<int> qi;
	mutex m_mtx;
	condition_variable m_cv_not_empty;
};
```



## 多个生产者多个消费者

```c++
//多个生产者，多个消费者
class QueueMoreConsumerMoreProducer 
{
public:
    QueueMoreConsumerMoreProducer(int capacity) 
    {
        m_capacity = capacity;
    }
    
    void enqueue(int element)
    {
        unique_lock<mutex> lck(m_mtx);
        // 等待队列非满
         while(m_capacity == m_queue.size()) // full
         {
             m_cv_not_full.wait(lck); 
        }
        m_queue.push(element);
        m_cv_not_empty.notify_one(); // 通知队列非空
    }
    
    int dequeue() 
    {
        unique_lock<mutex> lck(m_mtx);
        // 等待队列非空
         while(m_queue.empty())
         {
             m_cv_not_empty.wait(lck); 
         }
        int element = m_queue.front();
        m_queue.pop();
        m_cv_not_full.notify_one(); // 通知队列非满
        return element;
    }
    
    int size() 
    {
        unique_lock<mutex> lck(m_mtx);
        return m_queue.size();
    }

private:
    queue<int> m_queue;
    int m_capacity;
    mutex m_mtx;
    condition_variable m_cv_not_empty;
    condition_variable m_cv_not_full;

};
```

# 排序

## 时间复杂度

**O**(1) > **O** (logn) > **O** (**n**) > **O** (**nlogn**).

![image-20210705082249384](https://i.loli.net/2021/07/05/91QIleXfYCnRyzG.png)

## 排序算法时间复杂度

![image-20210704215738138](https://i.loli.net/2021/07/25/LwCDd5QK1EyXu3q.png)

## 堆排序

https://en.cppreference.com/w/cpp/container/priority_queue

https://www.bilibili.com/video/BV1Eb41147dK?from=search&seid=8324476198354549114

1. 完全二叉树
2. parent>children

堆是具有以下性质的完全二叉树：每个结点的值都大于或等于其左右孩子结点的值，称为大顶堆；

或者每个结点的值都小于或等于其左右孩子结点的值，称为小顶堆。如下图：

![img](https://i.loli.net/2021/08/18/PfqKRbZv7wspG6H.png)

**大顶堆：arr[i] >= arr[2i+1] && arr[i] >= arr[2i+2]**  

**小顶堆：arr[i] <= arr[2i+1] && arr[i] <= arr[2i+2]**  

![image-20210818142848571](https://i.loli.net/2021/08/18/7Lnj8GcMbwY4Aza.png)



## 归并排序

https://leetcode-cn.com/problems/sort-an-array/

```c++
class Solution {
	vector<int> tmp;
	void mergeSort(vector<int>& nums, int left, int right) {
		//递归结束条件是只有一个数，不需要对其进行排序
		if (left >= right) return;
		int mid = left + (right - left) / 2;
		mergeSort(nums, left, mid);
		mergeSort(nums, mid + 1, right);
		int i = left, j = mid + 1;
		int cnt = 0;
		while (i <= mid && j <= right) {
			if (nums[i] <= nums[j]) {
				tmp[cnt++] = nums[i++];
			}
			else {
				tmp[cnt++] = nums[j++];
			}
		}
		while (i <= mid) {
			tmp[cnt++] = nums[i++];
		}
		while (j <= right) {
			tmp[cnt++] = nums[j++];
		}
		

		for (int i = 0; i < right - left + 1; ++i) {
			nums[i + left] = tmp[i];
		}
	}

public:
	vector<int> sortArray(vector<int>& nums) {
		//tmp是成员变量
		tmp.resize((int)nums.size(), 0);
		//左闭右闭，两边都是闭区间
		mergeSort(nums, 0, (int)nums.size() - 1);
		return nums;
	}
};
```



### 数组中的逆序对

https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/

```c++
class Solution {
public:
	int reversePairs(vector<int>& nums) {
		if (nums.size()<=1)
		{//数组中数字个数小于等于1，没有逆序对
			return 0;
		}

		vector<int> copynums = nums;
		vector<int> tmp(nums.size());
		return reversePairs(copynums,0, nums.size()-1,tmp);
	}
	
	int reversePairs(vector<int>& copynums,int left,int right, vector<int>& tmp)
	{
		//递归结束终止条件
		if (left>=right)
		{//只有一个数字，没有逆序对
			return 0;
		}
	
		int middleindex = left + (right - left) / 2;
		int leftpairs = reversePairs(copynums, left,middleindex,tmp);
		int righttpairs = reversePairs(copynums, middleindex+1,right, tmp);
		
		if (copynums[middleindex]<= copynums[middleindex+1])
		{
			return leftpairs + righttpairs;
		}
	
		int crosspairs = MergeCount(copynums,left, middleindex,right,tmp);
	
		return leftpairs + righttpairs + crosspairs;
	}
	
	int MergeCount(vector<int>& copynums, int left, int middle,int right, vector<int>& tmp)
	{
		for (int i = left;i<=right;++i)
		{
			tmp[i] = copynums[i];
		}
	
		//i和j分别指向第一个有序区间和第二个有序区间的首元素
		int i = left;
		int j = middle + 1;
	
		int count = 0;
		for (int k=left;k<=right;++k)
		{
			if (i == middle+1)
			{
				copynums[k] = tmp[j];
				++j;
			}
			else if (j == right+1)
			{
				copynums[k] = tmp[i];
				++i;
			}
			else if (tmp[i]<=tmp[j])
			{
				copynums[k] = tmp[i];
				++i;
			}
			else
			{
				copynums[k] = tmp[j];
				++j;
				//第一个有序数组的剩余元素
				count += (middle - i + 1);
			}
	
		}
	
		return count;
	}

};
```



## 快速排序

从所有数字的最左边或最右边选择一个数字作为基准数字，把它放在合适的位置上

对基准数字和所有其他数字依次进行顺序调整，把比它小的都放在一边，比它大的放在另一边  

对未处理的数字中两边的数字进行顺序调整，调整后把其中不是基准数字的数字排除在未处理范围之外

重复以上过程直到所有数字都被排除在未处理数字范围外。这个时候基准数字就被放在合适的位置上了

对左右两部分数字重复以上过程直到所有数字都被放在合适的位置上

### 方法1

```c++
void quick_sort(int *p_num, int size) {
    int base = *p_num, tmp = 0;
    int *p_start = p_num, *p_end = p_num + size - 1;
    if (size <= 1) {
        return ;
    }
    while (p_start < p_end) {
        if (*p_start > *p_end) {
            tmp = *p_start;
            *p_start = *p_end;
            *p_end = tmp;
        }
        if (*p_start == base) {
            //基准数字在前
            p_end--;
        }
        else {
            //基准数字在后
            p_start++;
        }
    }
    quick_sort(p_num, p_start - p_num);
    quick_sort(p_start + 1, p_num + size - 1 - p_start);
}
```



### 方法2

```c++
class Solution {
public:

	void QuickSort(vector<int>& nums, int left, int right)
	{
		if (left >= right)
		{
			return;
		}
        
        //当数组有序的时候，时间复杂度较高，可以随机选择一个作为主元
        //int index = rand() % (right - left + 1) + left; // 随机选一个作为我们的主元
        //swap(nums[left], nums[index]);
        
		int base = nums[left];
		int start = left, end = right;
	
		while (left < right)
		{
			if (nums[left] > nums[right])
			{
				swap(nums[left], nums[right]);
			}
	
			if (base == nums[left])
			{//基准数字在前面,说明最后一个数字比基准数字大，可以跳过
				right--;
			}
			else
			{//基准数字在后面,说明第一个数字比基准数字小，可以跳过
				left++;
			}
		}
		QuickSort(nums, start, left - 1);
		QuickSort(nums, right + 1, end);
	}
	
	vector<int> sortArray(vector<int>& nums) {
		QuickSort(nums, 0, nums.size() - 1);
		return nums;
	}

};
```

### 方法3

```c++
//快速排序
/* 快速排序主函数 */
void Quicksort(vector<int>& nums) {
	// 一般要在这用洗牌算法将 nums 数组打乱，
	// 以保证较高的效率，我们暂时省略这个细节
	sort(nums, 0, nums.size - 1);
}

/* 快速排序核心逻辑 */
void sort(vector<int>& nums, int lo, int hi) {
	if (lo >= hi) return;
	// 通过交换元素构建分界点索引 p
	int p = partition(nums, lo, hi);
	// 现在 nums[lo..p-1] 都小于 nums[p]，
	// 且 nums[p+1..hi] 都大于 nums[p]
	sort(nums, lo, p - 1);
	sort(nums, p + 1, hi);
}

int partition(vector<int>& nums, int lo, int hi) {
	if (lo == hi) return lo;
	// 将 nums[lo] 作为默认分界点 pivot
	int pivot = nums[lo];
	// j = hi + 1 因为 while 中会先执行 --
	int i = lo, j = hi + 1;
	while (true) {
		// 保证 nums[lo..i] 都小于 pivot
		while (nums[++i] < pivot) {
			if (i == hi) break;
		}
		// 保证 nums[j..hi] 都大于 pivot
		while (nums[--j] > pivot) {
			if (j == lo) break;
		}
		if (i >= j) break;
		// 如果走到这里，一定有：
		// nums[i] > pivot && nums[j] < pivot
		// 所以需要交换 nums[i] 和 nums[j]，
		// 保证 nums[lo..i] < pivot < nums[j..hi]
		swap(nums, i, j);
	}
	// 将 pivot 值交换到正确的位置
	swap(nums, j, lo);
	// 现在 nums[lo..j-1] < nums[j] < nums[j+1..hi]
	return j;
}

// 交换数组中的两个元素
void swap(int[] nums, int i, int j) {
	int temp = nums[i];
	nums[i] = nums[j];
	nums[j] = temp;
}


int main()
{
	return 0;
}
```

### 方法4

```C++
class Solution {
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }
    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l; // 随机选一个作为我们的主元
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }
    void randomized_quicksort(vector<int>& nums, int l, int r) {
        if (l < r) {
            int pos = randomized_partition(nums, l, r);
            randomized_quicksort(nums, l, pos - 1);
            randomized_quicksort(nums, pos + 1, r);
        }
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        srand((unsigned)time(NULL));
        randomized_quicksort(nums, 0, (int)nums.size() - 1);
        return nums;
    }
};
```



## 插入排序

 把没有排好序的数字中最前面或最后面的数字插入到排好序的数字中合适的位置上，重复这个过程直到把所有数字都放在合适的位置上 

每次把要插入的数字和它前面或后面的数字进行顺序调整，重复这个过程直到它被放在合适的位置上

```c++
void insert_sort(int *p_num, int size) 
{
	int num = 0, num1 = 0, tmp = 0;
	for (num = 1; num <= size - 1; num++) {
		//把下标为num的数字插入到前面合适的位置上
		for (num1 = num - 1; num1 >= 0; num1--) {
			//调整下标为num1和num1+1存储区里数字的顺序
			if (*(p_num + num1) > *(p_num + num1 + 1)) {
				tmp = *(p_num + num1);
				*(p_num + num1) = *(p_num + num1 + 1);
				*(p_num + num1 + 1) = tmp;
			}
			else {
				break;
			}
		}
	}
}
```

```c++
void insertion_sort(vector<int> &nums, int n) 
{
	for (int i = 0; i < n; ++i) 
	{
		for (int j = i; j > 0 && nums[j] < nums[j - 1]; --j) 
		{
			swap(nums[j], nums[j - 1]);
		}
	}
}
```



## [数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```c++
 int findKthLargest(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end(), [](int a, int b) {return a > b; });
	    return nums[k - 1];
    }
```

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int findindex = nums.size() - k;
	    return QuickSort(nums,0,nums.size()-1,findindex);
    }

    int QuickSort(vector<int>& nums,int left,int right,int findindex)
    {
        if (left>=right && left  == findindex)
        {
            return nums[left];
        }
        //快速排序是确定一个数字的下标，左边的比之小，右边的比之大
        int base = nums[left];
        int start = left, end = right;
        while (left<right)
        {
            if (nums[left]>nums[right])
            {
                swap(nums[left],nums[right]);
            }
    
            if (base == nums[left])
            {
                right--;
            }
            else if (base = nums[right])
            {
                left++;
            }
        }
    
        if (left == findindex)
        {
            return nums[left];
        }
        else
        {
            return left > findindex ? QuickSort(nums, start, left - 1, findindex) : QuickSort(nums, left + 1, end, findindex);
        }
    }

};
```



```c++
class Solution {
public:

    void heapify(vector<int>& vi, int heapsize, int i)
    {
        if (i >= heapsize)
        {
            return;
        }
        //第一个子节点和第二个子节点的下标
        int c1 = 2 * i + 1, c2 = 2 * i + 2;
        int max = i;
        if (c1 < heapsize && vi[max] < vi[c1])
        {
            max = c1;
        }
        if (c2 < heapsize && vi[max] < vi[c2])
        {
            max = c2;
        }
    
        if (max != i)
        {
            swap(vi[max], vi[i]);
            heapify(vi, heapsize, max);
        }
    }
    
    void build_heap(vector<int>& vi, int heapsize)
    {
        int last_node = heapsize - 1;
        int parent = (last_node - 1) / 2;
        for (int i = parent; i >= 0; --i)
        {
            heapify(vi, heapsize, i);
        }
    }
    
    int findKthLargest(vector<int>& nums, int k) {
        int heapSize = nums.size();
        build_heap(nums, heapSize);
        for (int i = nums.size() - 1; i >= nums.size() - k + 1; --i) {
            swap(nums[0], nums[i]);
            --heapSize;
            heapify(nums,heapSize,0);
        }
        return nums[0];
    }


};
```

```c++
int findKthLargest1(vector<int>& nums, int k)
{
	priority_queue<int> pqi;
	for (auto num:nums)
	{
		pqi.push(num);
	}

	for(int i=0;i<k-1;++i)
	{
		pqi.pop();
	}
	return pqi.top();

}
```

## [ 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

注：priority_queue自定义函数的比较与sort正好是相反的，也就是说，如果你是把大于号作为第一关键字的比较方式，那么堆顶的元素就是第一关键字最小的

```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> counts;
        int max_count = 0;
        for (auto num:nums)
        {
            max_count = max(max_count,++counts[num]);
        }

        vector<vector<int>> buckets(max_count+1);
        for (auto count:counts)
        {
            buckets[count.second].push_back(count.first);
        }
    
        vector<int> res;
        for (int i=max_count;i>=0 && res.size()<k;--i)
        {
            for (auto num: buckets[i])
            {
                res.push_back(num);
                if (res.size() == k)
                {
                    break;
                }
            }
        }
    
        return res;
    }

};
```

```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> counts;
        for (auto num : nums)
        {
            counts[num]++;
        }

        //priority_queue自定义函数的比较与sort正好是相反的，也就是说，如果你是把大于号作为第一关键字的比较方式，那么堆顶的元素就是第一关键字最小的
        auto cmp = [](const pair<int, int>& pa, const pair<int, int> &pb) {
            return pa.second > pb.second; };
        //decltype获取表达式的数据类型
        //构建小堆
        //std::priority_queue<int, std::vector<int>, std::greater<int>> q2(data.begin(), data.end());
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
    
        for (auto count:counts)
        {
            if (q.size() == k)
            {
                if (q.top().second<count.second)
                {
                    q.pop();
                    q.emplace(count.first, count.second);
                }
            }
            else
            {
                q.emplace(count.first,count.second);
            }
        }
    
        vector<int> res;
        while (!q.empty())
        {
            res.emplace_back(q.top().first);
            q.pop();
        }
    
        return res;
    }

};
```



# 查找

## lower_bound

## upper_bound

//注意：当数组中没有要查找的元素时，lower_bound和upper_bound返回的结果相同

```c++
//注意：当数组中没有要查找的元素时，lower_bound和upper_bound返回的结果相同
std::vector<int> vi{ 1,2,3,3,3,3,4,5,6 };
// Search for first element x such that target ≤ x,左边界
auto lower = std::lower_bound(vi.begin(), vi.end(), 3);
if (lower == vi.end())
{
    std::cout << "Not Found!\n";
}
else
{
    std::cout << "*lower:" << *lower << std::endl;
    std::cout << "index:" << lower-vi.begin() << std::endl;
}

//Search first element that is greater than target
auto upper = std::upper_bound(vi.begin(), vi.end(), 3);
if (upper == vi.end())
{
    std::cout << "Not Found!\n";
}
else
{
    std::cout << "*upper:" << *upper << std::endl;
    std::cout << "index:" << upper-1 - vi.begin() << std::endl;
}

```



## 二分查找

时间复杂度O(logN)

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485044&idx=1&sn=e6b95782141c17abe206bfe2323a4226&scene=21#wechat_redirect

```c++
//二分查找
//时间复杂度:O(logN)
int FindTarget(const vector<int>vi,int target)
{
	if (vi.size()<=0)
	{
		return -1;
	}
    //左闭右闭
	int left = 0, right = vi.size() - 1;
	
	while (left<=right)
	{
		//int middleindex = (left + right) / 2;
		//left + right有可能溢出，化简后两种写法一致
		int middleindex = left + (right - left) / 2;
		if (vi[middleindex] == target)
		{
			return middleindex + 1;
		}
		else if (vi[middleindex] > target)
		{
			right = middleindex - 1;
		}
		else
		{
			left = middleindex + 1;
		}
	
	}


	return -1;

}
```



## 求开方

https://leetcode-cn.com/problems/sqrtx/

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

```c++
int mySqrt(int x)
{
	int res = -1;
	int left = 0, right = x;
	while (left<=right)
	{
		int middle = left + (right - left) / 2;
		//需要转为long类型，否则有可能溢出
		if ((long)middle*middle <= x)
		{
			res = middle;
			left = middle + 1;
		}
		else
		{
			right = middle - 1;
		}
	}
	return res;
}
```





## 在排序数组中查找元素的第一个和最后一个位置

https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。



```c++
std::vector<int> searchRange(std::vector<int>& nums, int target)
{
	//<algorithm>
	 // Search for first element x such that target ≤ x,左边界
	auto left = lower_bound(nums.begin(), nums.end(), target);
	// Search first element that is greater than target（指向target元素的下一个元素）
	auto right = upper_bound(nums.begin(), nums.end(), target);

	//if (left == nums.end() || right == nums.end())
	//如果数组中只有一个元素时并且等于taget,upper_bound函数的返回结果时nums.end()
	//如果数组中没有target时，lower_bound和upper_bound函数返回的结果相同
	if (left == right)
	{
		return { -1,-1 };
	}
	
	return { int(left - nums.begin()) ,int(right - nums.begin() - 1) };

}
```



```c++
class Solution {
public:

    //寻找左边界

int left_bound(std::vector<int>& nums, int target)
{
	int left = 0,right = nums.size() - 1;
	int res = -1;
	while (left<=right)
	{
		int middle = left + (right-left)/2;
		if (nums[middle] == target)
		{
            //right = middle
			right = middle-1;
			res = middle;
		}
		else if (nums[middle] > target)
		{
			right = middle - 1;
		}
		else if (nums[middle] < target)
		{
			left = middle + 1;
		}
	}

	return res;

}

int right_bound(std::vector<int>& nums, int target)
{
	int left = 0, right = nums.size() - 1;
	int res = -1;
	while (left <= right)
	{
		int middle = left + (right - left) / 2;
		if (nums[middle] == target)
		{
            //left = middle
			left = middle+1;
			res = middle;
		}
		else if (nums[middle] > target)
		{
			right = middle - 1;
		}
		else if (nums[middle] < target)
		{
			left = middle + 1;
		}
	}

	return res;

}


    vector<int> searchRange(vector<int>& nums, int target) {
    int left = left_bound(nums,target);
    int right = right_bound(nums, target);
    if(left>=0 && right<= nums.size()-1 && nums[left] == target && nums[right] == target)
    {
    	return {left,right};
    }
    else
    {
    	return {-1,-1};
    }
    }

};
```



## 搜索旋转排序数组


整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

https://leetcode-cn.com/problems/search-in-rotated-sorted-array/

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(!nums.size())
        {
            return -1;
        }

        if(nums.size() == 1)
        {
            return nums[0] == target? 0:-1;
        }

        int size = nums.size();
        int left =0,right = size-1;
        while(left<=right)
        {
            int middle = left+(right-left)/2;
            if(nums[middle] == target)
            {
                return middle;
            }
            if(nums[left]<=nums[middle])
            {//前半部分有序
                if(target>=nums[left] && target<nums[middle])
                {
                    right = middle-1;
                }
                else{
                    left=middle+1;
                }
            }
            else
            {//后半部分有序
                if(target>nums[middle] && target<=nums[right])
                {
                    left=middle+1;
                }
                else{
                    right = middle-1;
                }
            }

        }

        return -1;
    }
};
```



## 搜索旋转排序数组（有重复数字）

https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/

```c++
bool search(vector<int>& nums, int target) {
	if (nums.empty())
	{
		return false;
	}
	if (nums.size()==1)
	{
		return nums[0] == target;
	}

	int left = 0, right = nums.size() - 1;
	while (left<=right)
	{
		int middle = left + (right-left) / 2;
		if (nums[middle] == target)
		{
			return true;
		}
	
		if (nums[middle] == nums[left])
		{
			left++;
		}
		else if (nums[middle] > nums[left])
		{//middle在第一个有序数组中
			//nums[left] <= target，如果target<nums[left],可能会错过
			if (nums[middle]>target &&nums[left] <= target)
			{
				right = middle - 1;
			}
			else
			{
				left = middle + 1;
			}
		}
		else 
		{//middle在第二个有序数组中
			//nums[right]>=target，如果target>nums[right],可能会错过
			if (nums[middle] < target && nums[right]>=target)
			{
				left = middle + 1;
				
			}
			else
			{
				right = middle - 1;
			}
		}
	}
	
	return false;

}
```



## 代码模板

```
int binary_search(int[] nums, int target) {
    int left = 0, right = nums.length - 1; 
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1; 
        } else if(nums[mid] == target) {
            // 直接返回
            return mid;
        }
    }
    // 直接返回
    return -1;
}

int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，锁定左侧边界
            right = mid - 1;
        }
    }
    // 最后要检查 left 越界的情况
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}

int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，锁定右侧边界
            left = mid + 1;
        }
    }
    // 最后要检查 right 越界的情况
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
```



# 字符串

## 请按长度为8拆分每个字符串后输出到新的字符串数组；

题目描述：

•连续输入字符串，请按长度为8拆分每个字符串后输出到新的字符串数组；
•长度不是8整数倍的字符串请在后面补数字0，空字符串不处理。

### 输入描述：

连续输入字符串(输入多次,每个字符串长度小于100)

### 输出描述：

输出到长度为8的新字符串数组

### 示例1

输入：

```
abc
123456789
```

输出：

```
abc00000
12345678
90000000
```

```c++
#include <iostream>

using namespace std;

int main()
{
    string str;
    //连续输入字符串(输入多次)
    // while(cin >> s)
    while(getline(cin,str))
    {
        //str的长度大于8
        while(str.size()>8)
        {
            /*
            substr(size_type pos = 0, size_type count = npos),
            pos:position of the first character to include
            count:length of the substring
            */
            cout<<str.substr(0,8)<<endl;
            //当count为默认值时，默认将剩下字符串全部返回
            str=str.substr(8);
        }
        //append( size_type count, CharT ch )
        cout<<str.append(8-str.size(),'0')<<endl;
    }
    return 0;
}
```

## 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。



### 思路

使用unordered_map实现，key为字符，value为字符出现的次数

不能使用set，存在删除，只适合于成对出现的字符串

unordered_map存储的元素无序，想要找到s中第一个只出现一次的字符，不能遍历unordered_map，遍历s字符串即可

### 代码

```
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char,int> dic;
        for(auto ch:s)
        {
            dic[ch]++;   
        }
        for(auto ch:s)
        {
            if(dic[ch]== 1)
            {
                 return ch;
             }
        }
        return ' ';
    }
};
```





# 树

## 二叉搜索树与双向链表

剑指 Offer 36.

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

### 思路

二叉搜索树有序，使用中序遍历

// 打印中序遍历
void dfs(Node* root) {
    if(root == nullptr) return;
    dfs(root->left); // 左
    cout << root->val << endl; // 根
    dfs(root->right); // 右
}

### 代码

```c++
class Solution {

public:

  Node* treeToDoublyList(Node* root) {
​    if(!root)
​      return nullptr;
​    dfs(root);
​    head->left = pre;
​    pre->right = head;
​    return head;
  }

  Node *pre=nullptr, *head=nullptr;
  void dfs(Node* cur)
  {
​    if(!cur)
​      return;
​     dfs(cur->left);
​    if(pre)
​      pre->right = cur;
​    else
​      head = cur;
​    cur->left = pre;
​    pre = cur;
​    dfs(cur->right);
  }
};
```





##  从上到下打印二叉树

剑指 Offer 32 - I.

### 思路

使用层次遍历

### 代码

```c++
//层次遍历基本流程
vector<int> levelOrder(TreeNode* root) {
        if(!root)
        {
            return {};
        }
        vector<int> res;
        queue<TreeNode*> qtree;
        qtree.push(root);
        while(!qtree.empty())
        {
           TreeNode* node =  qtree.front();
           qtree.pop();
           res.push_back(node->val);
           if(node->left)
           {
               qtree.push(node->left);
           }
           if(node->right)
           {
               qtree.push(node->right);
           }
        }
        return res;
    }


```



##  从上到下打印二叉树 II

剑指 Offer 32 - II.

### 思路

使用层次遍历

加一个size判断

### 代码

```
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q;
        vector<vector<int> > ans;
        if(root==NULL){
            return ans;
        }
        q.push(root);
        while(!q.empty()){
            vector<int> temp;
            for(int i=q.size();i>0;i--){
                TreeNode* node = q.front();
                q.pop();
                temp.push_back(node->val);
                if(node->left!=NULL) q.push(node->left);
                if(node->right!=NULL) q.push(node->right);
            }
            ans.push_back(temp);
        }

        return ans;
    }
};
```



## 从上到下打印二叉树 III

剑指 Offer 32 - III. 

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

### 思路

（1）根据层次的奇偶性翻转vector

（2）使用两个stack

为了达到这样打印的效果，我们需要使用两个栈。我们在打印某一行结点时，把下一层的子结点保存到相应的栈里。如果当前打印的是奇数层（第一层、第三层等），则先保存左子树结点再保存右子树结点到第一个栈里。如果当前打印的是偶数层（第二层、第四层等），则则先保存右子树结点再保存左子树结点到第二个栈里。

### 代码

```c++
class Solution {
public:
  vector<vector<int>> levelOrder(TreeNode* root) {
​    queue<TreeNode*> q;
​    vector<vector<int> > ans;
​    if(root==NULL){
​      return ans;
​    }
​    q.push(root);
​    while(!q.empty()){
​      vector<int> temp;
​      for(int i=q.size();i>0;i--){
​        TreeNode* node = q.front();
​        q.pop();
​        temp.push_back(node->val);
​        //cout<<node->val<<endl;
​        if(node->left!=NULL) q.push(node->left);
​        if(node->right!=NULL) q.push(node->right);
​      }
​      if(ans.size()%2)
​      {
​         reverse(temp.begin(),temp.end());
​        ans.push_back(temp);
​      }
​      else
​      {
​        ans.push_back(temp);
​      }
​    }
​    return ans;
  }

};
```

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int> > result;
        if(pRoot == NULL){
            return result;
        }
        stack<TreeNode* > s[2];
        s[0].push(pRoot);
        while(!s[0].empty() || !s[1].empty()){
            vector<int> v[2];
            // 偶数行
            while(!s[0].empty()){
                v[0].push_back(s[0].top()->val);
                if(s[0].top()->left != NULL){
                    s[1].push(s[0].top()->left);
                }
                if(s[0].top()->right != NULL){
                    s[1].push(s[0].top()->right);
                }
                s[0].pop();
            }
            if(!v[0].empty()){
                result.push_back(v[0]);
            }
            // 奇数行
            while(!s[1].empty()){
                v[1].push_back(s[1].top()->val);
                if(s[1].top()->right != NULL){
                    s[0].push(s[1].top()->right);
                }
                if(s[1].top()->left != NULL){
                    s[0].push(s[1].top()->left);
                }
                s[1].pop();
            }
            if(!v[1].empty()){
                result.push_back(v[1]);
            }
        }
        return result;
    }
};
```



##  二叉搜索树的后序遍历序列

剑指 Offer 33

### 思路

二叉搜索树的后续遍历最后一个节点一定是根节点（左右根）

比根节点小的节点为左子树，比根节点大的节点为右子树

左右子树也符合二叉搜索树的后续遍历规律

### 代码

```c++
class Solution {

public:

  bool verifyPostorder(vector<int>& postorder) {
​    return Recur(postorder,0,postorder.size()-1);
  }
  bool Recur(vector<int>& postorder,int start,int end)
  {
​    //当只有一个节点的时候，start=end，肯定是后序遍历
​    //当只有两个节点的时候，start=end+1,肯定是后序遍历
​    //当左子树或者右子树为空的时候start>end
​    if(start>=end)
​    {
​      return true;
​    }
​    int rootvalue = postorder[end];
​    int i=start;
​    while(postorder[i]<rootvalue)
​    {
​      ++i;
​    }
​    int leftend = i-1;
​    while(postorder[i]>rootvalue)
​    {
​      ++i;
​    }
​    return i==end && Recur(postorder,start,leftend) && Recur(postorder,leftend+1,end-1);
  }
};
```





## 二叉树的最大深度

```c++
class Solution {
public:
     //1 明确函数功能，计算二叉树的最大深度
	int maxDepth(TreeNode* root) {
		//2 寻找递归结束条件
		if (!root)
		{
			return 0;
		}
		int leftheight = maxDepth(root->left);
		int rightheight = maxDepth(root->right);

		//3 最大深度 = max{左子树深度，右子树深度}+1
		//max函数在windows.h中定义
		return max(leftheight, rightheight)+1;
	}

};
```



## 二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回它的 **中序** 遍历。

### 递归

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root)
        {
            return {};
        }
        vector<int> vi;
        InOrder(root,vi);
        return vi;
    }

    void InOrder(TreeNode* root,vector<int>& vi)
    {//中序遍历，左根右
        if(!root)
        {
            return;
        }
        InOrder(root->left,vi);
        vi.push_back(root->val);
        InOrder(root->right,vi);
    }

};
```



### 非递归

```C++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root)
        {
            return {};
        }
        vector<int> vi;
        stack<TreeNode*> si;
        while(root || !si.empty())
        {
            while(root)
            {
                si.push(root);
                root = root->left;
            }

            TreeNode*  tmp = si.top();
            si.pop();
            vi.push_back(tmp->val);
            root= tmp->right;
        }
    
        return vi;
    }

};
```



## 二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回它的 **中序** 遍历。

### 递归

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root)
        {
            return {};
        }
        vector<int> vi;
        InOrder(root,vi);
        return vi;
    }

    void InOrder(TreeNode* root,vector<int>& vi)
    {//中序遍历，左根右
        if(!root)
        {
            return;
        }
        InOrder(root->left,vi);
        vi.push_back(root->val);
        InOrder(root->right,vi);
    }

};
```



### 非递归

```C++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root)
        {
            return {};
        }
        vector<int> vi;
        stack<TreeNode*> si;
        while(root || !si.empty())
        {
            while(root)
            {
                si.push(root);
                root = root->left;
            }

            TreeNode*  tmp = si.top();
            si.pop();
            vi.push_back(tmp->val);
            root= tmp->right;
        }
    
        return vi;
    }

};
```

# 链表

## 复杂链表的复制

剑指 Offer 35

### 思路

使用map存储旧节点和新节点的对应关系

### 代码

```c++
class Solution {

public:

  Node* copyRandomList(Node* head) {

    if(!head)
    
    {
    
      return nullptr;
    
    }
    
    Node *reslist = new Node(-1);
    
    Node *reshead = reslist;

  




     map<Node *,Node *> mapnode;
    
     Node* head1 =head;



  

    while(head1)
    
    {
    
      Node *node = new Node(head1->val);

  


      reslist->next = node;


​      

      mapnode.insert({head1,node});


​      

      reslist = reslist->next;
    
      head1 = head1->next;
    
    }





    head1 =head;



    while(head1)
    
    {
    
      Node *newnode = mapnode[head1];
    
      newnode->random = mapnode[head1->random];
    
      head1 = head1->next;
    
    }

  


     return reshead->next;

  }

};
```



```c++
class Solution {

public:

  Node* copyRandomList(Node* head) {

    if(!head)
    
    {
    
      return nullptr;
    
    }
    
    //key为当前节点，value为复制节点
    
    map<Node*,Node*> hashmapnodes;



    Node* node = head;



    Node* newhead = new Node(node->val);
    
    hashmapnodes.insert({node,newhead});



    Node* next = node->next;
    
    while(next)
    
    {
    
      Node* tmp = new Node(next->val);
    
      hashmapnodes.insert({next,tmp});
    
      next = next->next;
    
    }



    Node* head1 = head;



    // while(head1)
    
    // {
    
    //   Node *newnode = hashmapnodes[head1];
    
    //   if(head1 &&head1->random)
    
    //   {
    
    //     newnode->random = hashmapnodes[head1->random];
    
    //   }
    
    //   if(head1 &&head1->next)
    
    //   newnode->next = hashmapnodes[head1->next];



    //   head1 = head1->next;
    
    // }



    for(auto n:hashmapnodes)
    
    {
    
      Node *newnode = hashmapnodes[n.first];
    
      newnode->random = hashmapnodes[n.first->random];
    
      newnode->next = hashmapnodes[n.first->next];
    
    }


​    

    return newhead;

  }

};
```



# 数组

## 数组中重复的数字

### 方法1 先排序后遍历，时间复杂度为O(Nlog(N))，空间复杂度为o(1)

```c++
/*
数组中重复的数据
给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。

找到所有出现两次的元素。

你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void FindRepeatNum(vector<int> &vi)
{
    //默认从小到大排序，时间复杂度为O(Nlog(N))
	sort(vi.begin(),vi.end());
	for (int i=0;i< vi.size();++i)
	{
		if (i+1< vi.size() && vi[i] == vi[i+1])
		{
			cout << i << endl;
			i++;
		}
	}
}



int main()
{
    std::cout << "Hello World!\n";
}
```

### 方法2 顺序遍历，使用哈希表记录重复数字，时间复杂度为o(n),空间复杂读为o(n)

```c++
class Solution {
public:
  int findRepeatNumber(vector<int>& nums) {
​    unordered_set<int> si;
​    for(auto num:nums)
​    {
​      if(si.count(num))
​      {
​        return num;
​      }
​      si.insert(num);
​    }
​    return -1;
  }
};
```

### 方法三 原地排序，时间复杂读为o(n)，空间复杂度为o(1)

```c++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        for(size_t i=0;i<nums.size();++i)
        {
            while(nums[i]!=i)
            {//虽然是双重循环，但每次都会将一个元素放到属于它的位置上
                if(nums[i] == nums[nums[i]])
                {
                    return nums[i];
                }
                swap(nums[i],nums[nums[i]]);
            }
        }
        return -1;
    }
};
```

# 玩转双指针

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484505&idx=1&sn=0e9517f7c4021df0e6146c6b2b0c4aba&chksm=9bd7fa51aca07347009c591c403b3228f41617806429e738165bd58d60220bf8f15f92ff8a2e&scene=21#wechat_redirect



https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485141&idx=1&sn=0e4583ad935e76e9a3f6793792e60734&scene=21#wechat_redirect

## 左右指针

两个指针指向同一个数组，但是遍历方向相反，则可以用来进行搜索，待搜索的数组往往是排好序的

主要解决数组（或者字符串）中的问题，比如二分查找

左右指针在数组中实际是指两个索引值，一般初始化为 left = 0, right = nums.length - 1 

### 1 二分查找

https://leetcode-cn.com/problems/binary-search/

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if (nums.size()<=0)
        {
            return -1;
        }
        int left = 0, right = nums.size() - 1;
        

        while (left<=right)
        {
            int middleindex = (left + right) / 2;
            if (nums[middleindex] == target)
            {
                return middleindex;
            }
            else if (nums[middleindex] > target)
            {
                right = middleindex - 1;
            }
            else
            {
                left = middleindex + 1;
            }
    
        }


        return -1;
    }

};
```



### 2 两数之和

https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/submissions/

只要数组有序，就应该想到双指针技巧

```c++
vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() - 1;
	while (left<right)
	{
		int sum = numbers[left] + numbers[right];
		if (sum == target)
		{
            //下标从1开始，所有左右index都需要加1
			return { left +1,right+1};
		}
		else if (sum > target)
		{
			right--;
		}
		else if (sum < target)
		{
			left++;
		}
	}

	return {};
	}
```



### 3 反转数组

```java
void reverse(int[] nums) {
    int left = 0;
    int right = nums.length - 1;
    while (left < right) {
        // swap(nums[left], nums[right])
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
        left++; right--;
    }
}
```



## 快慢指针

主要解决链表中的问题

### 1 判定链表中是否含有环

https://leetcode-cn.com/problems/linked-list-cycle/

```c++
 bool hasCycle(ListNode *head) {
        ListNode *slow = head;
		ListNode *fast = head;
        //只需要判断fast以及fast->next即可，slow指针不需要判断
		while (fast && fast->next)
		{
			fast = fast->next->next;
			slow = slow->next;
			if (fast == slow)
			{
				return true;
			}
		}

		return false;
	}
```



### 2 已知链表中含有环，返回这个环的起始位置

https://leetcode-cn.com/problems/linked-list-cycle-ii/

```c++
 ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head;
		ListNode *fast = head;
		while (fast && fast->next)
		{
			fast = fast->next->next;
			slow = slow->next;
			if (fast == slow)
			{
				slow = head;
		        while (slow!=fast)
		        {
			        slow = slow->next;
			        fast = fast->next;
		        }

	    	    return slow;
			}
		}
	
		return nullptr;
	}
```



### 3 寻找链表的中点

类似上面的思路，我们还可以让快指针一次前进两步，慢指针一次前进一步，当快指针到达链表尽头时，慢指针就处于链表的中间位置。



```
ListNode slow, fast;
slow = fast = head;
while (fast != null && fast.next != null) {
    fast = fast.next.next;
    slow = slow.next;
}
// slow 就在中间位置
return slow;
```



当链表的长度是奇数时，slow 恰巧停在中点位置；如果长度是偶数，slow 最终的位置是中间偏右：



### 4 寻找链表的倒数第K个元素

 https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/submissions/

```c++
ListNode* getKthFromEnd(ListNode* head, int k) {
        if(!head)
        {
            return nullptr;
        }

        ListNode* fast =  head;
        ListNode* slow = head;
        for(int i=0;i<k;++i)
        {
            fast = fast->next;
        }
    
        while(fast)
        {
            fast = fast->next;
            slow = slow->next;
        }
    
        return slow;
    
    }


```

## 滑动窗口

若两个指针指向同一个数组，遍历方向相同且不会相交，则也成为滑动窗口（两个指针包围的区域即为当前的窗口），经常用于区间搜索。



### 算法框架

```c++
/* 滑动窗口算法框架 */
void slidingWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0; 
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 右移窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...
    
        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/
    
        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 左移窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }

}
```



### 1 最小覆盖字串

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。



### 2 字符串的排列

给你两个字符串 `s1` 和 `s2` ，写一个函数来判断 `s2` 是否包含 `s1` 的排列。

换句话说，`s1` 的排列之一是 `s2` 的 **子串** 。



### 3 找到字符串中所有字母的异位词

https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指字母相同，但排列不同的字符串



### 4  最长无重复子串

https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。



### 5 长度最小的子数组

https://leetcode-cn.com/problems/minimum-size-subarray-sum/solution/chang-du-zui-xiao-de-zi-shu-zu-by-leetcode-solutio/

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 



```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0, right = 0;
	int sum = 0;
	int res = INT_MAX;
	while (right<nums.size())
	{

		int rightdata = nums[right];
		right++;
	
		sum += rightdata;
	
		while (sum>=target)
		{
	        res = min(res,right-left);
			int leftdata = nums[left];
			left++;
	        sum-=leftdata;
				
		}
	
	}
	
	return res == INT_MAX?0:res;
	}

};
```



### 6 滑动窗口大小固定为n

```c++
bool checkInclusion(string s1, string s2) {
    unordered_map<char, int> need, window;
	for (auto s:s1)
	{
		need[s]++;
	}

	int valid = 0;
	int left = 0, right = 0;
	while (right< s2.size())
	{
		char c = s2[right];
		right++;
	
		if (need.count(c))
		{
			window[c]++;
			if (window[c] == need[c])
			{
				valid++;
			}
		}
	
	    /*** debug 输出的位置 ***/
	    printf("window: [%d, %d)\n", left, right);
	
		//滑动窗口大小固定为s1的大小
		while ((right-left)>=s1.size())
		{
			if (valid == need.size())
			{
				return true;
			}
	
			char d = s2[left];
			left++;

            //复制的时候需要修改c为d
    		if (need.count(d))
    		{
    			if (window[d] == need[d])
    			{
    				valid--;
    			}
    			window[d]--;
    		}
    	}
    }
    
    return false;
    }


```



## 归并两个有序数组

https://leetcode-cn.com/problems/merge-sorted-array/



```c++
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
int pos = m-- + n-- - 1;
while (m >= 0 && n >= 0) {
nums1[pos--] = nums1[m] > nums2[n]? nums1[m--]: nums2[n--];
}
while (n >= 0) {
nums1[pos--] = nums2[n--];
}
}
```



## 三数之和

```
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
       std::vector<std::vector<int>> res;
	if(nums.size()<3)
	{//如果数组没有三个元素，直接返回即可
		return {};
	}

	//排序，时间复杂的为0(nlogn)
	sort(nums.begin(),nums.end());

	int numslength = nums.size();


	for(int i=0;i<numslength;++i)
	{
		if(nums[i]>0)
		{//first number is bigger than zero,the behind number is bigger
			break;
		}

		if(i>0 && nums[i] == nums[i-1])
		{//去除重复数组，第二次遍历如果元素相同跳过即可
			continue;
		}

		//可以用双指针法，找到ｂ,c
		int left = i+1;
	  int right = numslength-1;

		while (left<right)
		{
			
			if(nums[i]+nums[left]+nums[right] == 0)
			{
				res.push_back({nums[i],nums[left],nums[right]});
				while (left<right && nums[left+1] == nums[left])
				{//去除重复解
					left++;
				}
				while (left<right && nums[right] == nums[right-1])
				{//去除重复解
					right--;
				}
				
				//需要注意，上面会将ｌｅｆｔ和ｒｉｇｈｔ移动到不相同的元素前面，需要再次进行移动
				left++;
				right--;
				
			}
			else if (nums[i]+nums[left]+nums[right]>0)
			{
				right--;
			}
			else
			{	
			  left++;
			}
		}
	
	}

	return res;
    }
};
```



# 回溯

## 算法模板

```text
def backtrack(...):
    for 选择 in 选择列表:
        做选择
        backtrack(...)
        撤销选择
```

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484709&idx=1&sn=1c24a5c41a5a255000532e83f38f2ce4&scene=21#wechat_redirect

https://zhuanlan.zhihu.com/p/93530380

## 全排列

https://leetcode-cn.com/problems/permutations-ii/

```cpp
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ret;
        sort(begin(nums), end(nums));
        do {
            ret.emplace_back(nums);
        } while (next_permutation(begin(nums), end(nums)));
        return ret;
    }
};
```



## 象棋从起点走到终点

1 只能走日字

2 只能向右上方走

输出返回从起点到终点的解法

```c++
#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;

int end_x, end_y;
int ans = 0;

void Test(int x,int y)
{
	if (x == end_x && y == end_y)
	{
		ans++;
		return;
	}
	if (x>end_x || y>end_y)
	{
		return;
	}

    //有两种选择
	x = x + 2;
	y = y + 1;
	Test(x,y);
	x = x - 2;
	y = y - 1;
	
	x = x + 1;
	y = y + 2;
	Test(x, y);
	x = x - 1;
	y = y - 2;

}

int main()
{
	int x0, y0, x1, y1;
	cin >> x0 >> y0 >> x1 >> y1;
	end_x = x1, end_y = y1;
	Test(x0, y0);
	cout << ans << endl; 
	return 0;
}
```





# 动态规划

1. 明确dp[n]代表的意义，需要返回什么（vector<int> vi(n+1,0);）
2. dp[0],dp[1]等赋值，对参数进行判断，可以直接返回
3. 明确状态转移方程
4. 一道题目中可以有多个dptable，最后求极值

## 一维

### [爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```c++
class Solution {
public:
	int climbStairs(int n) {
		if (n<= 1)
		{
			return 1;
		}
		if (n == 2)
		{
			return 2;
		}

		vector<int> vi(n+1,0);
		vi[1] = 1;
		vi[2] = 2;
		for (int i=3;i<=n;++i)
		{
			vi[i] = vi[i - 1] + vi[i - 2];
		}
		return vi[n];
	}

};
```



### [打家劫舍](https://leetcode-cn.com/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```c++
class Solution {
public:
	int rob(vector<int>& nums) {
		//dp[0]无意义，dp[1]表示抢劫第一家最大的金额，dp[n]表示抢劫第一家最大的金额
		vector<int> dp(nums.size()+1,0);
		dp[1] = nums[0];
		for(int i=2;i<=nums.size();++i)
		{ 
			dp[i] = max(dp[i-1],dp[i-2]+nums[i-1]);
		}

		return  dp[nums.size()];
	}

};
```



### [打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

```c++
class Solution {
public:
	//选择偷最后一间房子，第一个房子不能偷dp[n]
	//选择不偷最后一间房子，第一个房子可以偷dp[i-1]
	//返回两种情况的最大值
	//dp[0]没有意义
	int rob(vector<int>& nums) {
        if(nums.size() == 1)
        {
            return nums[0];
        }

		//1 明确dp[n]代表的意义
		vector<int> via(nums.size()+1,0);
		via[0] = 0;
		via[1] = 0;
		via[2] = nums[1];
		for (int i=3;i<=nums.size();++i)
		{
			//状态转移方程
			via[i] = max(via[i-1], via[i-2]+nums[i-1]);
		}
	
		vector<int> vib(nums.size() + 1, 0);
		vib[0] = 0;
		vib[1] = nums[0];
		for (int i=2;i<=nums.size()-1;++i)
		{
			vib[i] = max(vib[i-1], vib[i-2]+nums[i-1]);
		}
	
		return max(via[nums.size()], vib[nums.size()-1]);
	}

};
```



### [打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

```c++
class Solution {
public:
	//f表示节点被选中时，f(root)+g(left)+g(right)
	//g表示节点没有被选中，
	unordered_map<TreeNode*, int> f, g;
	//可以选择root，也可以不选择root
	int rob(TreeNode* root) {
		dfs(root);
		return max(f[root], g[root]);
	}

	void dfs(TreeNode* node)
	{
		if (!node)
		{
			return;
		}
		dfs(node->left);
		dfs(node->right);
	
		f[node] = node->val + g[node->left] + g[node->right];
		g[node] = max(f[node->left], g[node->left])+ max(f[node->right], g[node->right]);
	}

};
```



### [等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/)

如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。

例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。

子数组 是数组中的一个连续序列。

定义状态：dp[i]表示从nums[0]到nums[i]且以nums[i]为结尾的等差数列子数组的数量。

状态转移方程：dp[i] = dp[i-1]+1 if nums[i]-nums[i-1]==nums[i-1]-nums[i-2] else 0

解释：如果nums[i]能和nums[i-1]nums[i-2]组成等差数列，则以nums[i-1]结尾的等差数列均可以nums[i]结尾，且多了一个新等差数列[nums[i],nums[i-1],nums[i-2]]

```c++
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& nums) {
        int n = nums.size();
        if (n < 3) 
            return 0;
        vector<int> dp(n, 0);
        for (int i = 2; i < n; ++i)
        {
            if (nums[i] - nums[i-1] == nums[i-1] - nums[i-2]) {
                dp[i] = dp[i-1] + 1;
            }
        }
        return accumulate(dp.begin(), dp.end(), 0);
    }
};
```



## 二维

### [最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

```c++
class Solution {
public:
	int minPathSum(vector<vector<int>>& grid) {
		if (grid.empty())
		{
			return -1;
		}
		int m = grid.size();
		int n = grid[0].size();

		//
		vector<vector<int>> dp(m,vector<int>(n,0));
	
		dp[1][1] = grid[0][0];
	
		for (int i=0;i<m;++i)
		{
			for (int j=0;j<n;++j)
			{
				if (i==0 && j==0)
				{
					dp[i][j] = grid[i][j];
				}
				else if (i == 0)
				{
					dp[i][j] = dp[i][j - 1] + grid[i][j];
				}
				else if (j == 0)
				{
					dp[i][j] = dp[i-1][j] + grid[i][j];
				}
				else
				{
					//#include <algorithm>
					dp[i][j] = min(dp[i][j - 1], dp[i - 1][j])+ grid[i][j];
				}
			}
		}
	
		return dp[m-1][n-1];
	}

};
```

### [01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

```c++
class Solution {
public:
	//广度优先搜索
	vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
		if (mat.empty()) {
			return {};
		}
		int m = mat.size();
		int n = mat[0].size();
		vector<vector<int>> res(m,vector<int>(n,INT_MAX));
        vector<vector<bool>> visited(m,vector<bool>(n,false));
		queue<pair<int, int>> qi;
		for (int i=0;i<m;++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (mat[i][j] == 0)
				{
					res[i][j] = 0;
					qi.push({i,j});
				}
			}
		}

		vector<int> direction{-1,0,1,0,-1};
	    int level = 0;
		while (!qi.empty())
		{
	        level++;
			int size = qi.size();
			for (int i=0;i<size;++i)
			{
				auto[r, c] = qi.front();
				qi.pop();
	
				for (int j=0;j<4;++j)
				{//计算四个方向的距离
					int newx = r + direction[j];
					int newy = c + direction[j+1];
					if (newx>= 0 && newy>=0 && newx<m && newy <n)
					{
						if (res[newx][newy]!=0 && !visited[newx][newy])
						{
	                        //std::cout<<newx<<","<<newy<<std::endl;
							res[newx][newy] = level;
	                        visited[newx][newy] = true;
							qi.push({ newx ,newy});
						}
					}
				}
			}
		}
	
		return res;
	}

};
```

```c++
class Solution {
public:
	//dp table
	vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
		if (mat.empty()) {
			return {};
		}
		int m = mat.size();
		int n = mat[0].size();
		vector<vector<int>> res(m,vector<int>(n,INT_MAX-1));
		

		//水平向左，竖直向下
		for (int i=0;i<m;++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (mat[i][j] == 0)
				{
					res[i][j] = 0;
				}
				else
				{
					if (j>0)
					{
						res[i][j] = min(res[i][j], res[i][j - 1]+1);
					}
	
					if (i>0)
					{
						res[i][j] = min(res[i][j], res[i-1][j] + 1);
					}
					
				}
			}
		}
	
		//水平向右，竖直向上
		for (int i = m-1; i >=0; --i)
		{
			for (int j = n-1; j >=0; --j)
			{
				
				if (i <m-1)
				{
					res[i][j] = min(res[i][j], res[i+1][j] + 1);
				}
	
				if (j<n-1)
				{
					res[i][j] = min(res[i][j], res[i][j+1] + 1);
				}
	
			}
		}
	
		return res;
	}

};
```



### [统计全为 1 的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/)

给你一个 m * n 的矩阵，矩阵中的元素不是 0 就是 1，请你统计并返回其中完全由 1 组成的 正方形 子矩阵的个数。

```c++
class Solution {
public:
	int countSquares(vector<vector<int>>& matrix) {
		if (matrix.empty())
		{
			return 0;
		}

		int m = matrix.size();
		int n = matrix[0].size();
	    //dp[i][j]表示以i,j为右下角的最大正方形和正方形的数量
		vector<vector<int>> dp(m,vector<int>(n,0));
	    int res= 0;
		for (int i=0;i<m;++i)
		{
			for (int j=0;j<n;++j)
			{
				if (i==0 || j==0)
				{
					dp[i][j] = matrix[i][j];
				}
				else if (matrix[i][j] == 0)
				{
					dp[i][j] = 0;
				}
				else
				{
					dp[i][j] = min(min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
				}
	            res+=dp[i][j];
			}
		}
	
		return res;
	
	}

};
```

### [最大正方形](https://leetcode-cn.com/problems/maximal-square/)

在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。

```c++
class Solution {
public:
	int maximalSquare(vector<vector<char>>& matrix) {
		if (matrix.empty())
		{
			return 0;
		}

		int m = matrix.size();
		int n = matrix[0].size();
		vector<vector<int>> dp(m, vector<int>(n, 0));
	
		int res = -1;
	
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
	            //第一行和第一列进行区分
				if (i == 0 || j == 0)
				{
					if (matrix[i][j]=='0')
					{
						dp[i][j] = 0;
					}
					else
					{
						dp[i][j] = 1;
					}
				}
				else if (matrix[i][j] == '0')
				{
					dp[i][j] = 0;
				}
				else
				{
					dp[i][j] = min(min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i][j - 1]) + 1;
				}
	
				res = max(res, dp[i][j]);
			}
		}
		return res*res;
	}

};
```

## 分割类型

### [完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。



```c++
class Solution {
public:
	int numSquares(int n) {
		vector<int> dp(n+1,INT_MAX);

		dp[0] = 0;
		for (int i=1;i<=n;++i)
		{
			for (int j=1;j*j<=i;++j)
			{
				//i-j*j+    j*j = i
				//dp[i-j*j]+1
				dp[i] = min(dp[i],dp[i-j*j]+1);
			}
		}
	
		return dp[n];
	}

};
```



### [解码方法](https://leetcode-cn.com/problems/decode-ways/)

一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

```c++
class Solution {
public:
    int numDecodings(string s) {
        int size = s.size();
		//dp[i]表示以s[i]结尾的字符的解码方法数
		vector<int> dp(size+1,0);

		//增加dp[0]，方便计算dp[2]
		dp[0] = 1;
	
		//dp[n] = dp[n-1]+dp[n-2],类似青蛙跳台阶
		//1 2 3 4 5
		//dp[1]=1,以1结尾的字符的解码方法数
		//dp[2]=dp[1]+dp[0] = 1+1=2,以2结尾的字符的解码方法数(1,2和12两种)
		//dp[3]=dp[2]+dp[1] = 2+1=3,以3结尾的字符的解码方法数(1,2,3、12,3和1,23三种)
		//以此类推
	
		//base case
		if (s[0]!='0')
		{
			dp[1] = 1;
		}
		else
		{//字符串第一个有可能是0
			return 0;
		}
	
		for (int i=2;i<=size;++i)
		{
			//如果字符串中间0，只能与前一个字符进行拼接
			if (s[i - 1] != '0')
			{
				dp[i] += dp[i - 1];
			}
			//两个字符拼接后不能超过26
			if (s[i - 2]!='0' && (s[i-2]-'0')*10+(s[i-1]-'0')<=26)
			{
				dp[i] += dp[i - 2];
			}
			
		}
	
		return dp[size];
	}

}; 
```



### [单词拆分](https://leetcode-cn.com/problems/word-break/)

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

```
class Solution {
public:
	bool wordBreak(string s, vector<string>& wordDict) {
		int size = s.size();
		//dp[i]表示以字符s[i-1]结尾的字符串是否可以被拆分
		//dp[n] = dp[j] && s[j]...s[n-1]是否可以被拆分
		vector<bool> dp(size+1,false);
		//base case
		dp[0] = true;
		unordered_set<string> usetstr{ wordDict.begin(),wordDict.end()};

		for (int i=1;i<=size;++i)
		{
			for (int j=0;j<i;j++)
			{
	
				if (dp[j] && usetstr.count(s.substr(j,i-1-j+1)))
				{
					dp[i] = true;
					break;
				}
			}
		}
	
		return  dp[size];
	}

};
```



## 子序列问题

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484666&idx=1&sn=e3305be9513eaa16f7f1568c0892a468&chksm=9bd7faf2aca073e4f08332a706b7c10af877fee3993aac4dae86d05783d3d0df31844287104e&scene=21#wechat_redirect

```c++
/*
对于子序列问题，第一种动态规划方法是，定义一个 dp 数组，其中 dp[i] 表示以 i 结尾的子
序列的性质。在处理好每个位置后，统计一遍各个位置的结果即可得到题目要求的结果。

对于子序列问题，第二种动态规划方法是，定义一个 dp 数组，其中 dp[i] 表示到位置 i 为止
的子序列的性质，并不必须以 i 结尾。这样 dp 数组的最后一位结果即为题目所求，不需要再对每
个位置进行统计
*/
```



### [ 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```c++


class Solution {
public:
	//dp
	int lengthOfLIS(vector<int>& nums) {
		int size = nums.size();
		//dp[n]表示以nums[n]结尾的最长递增子序列长度
		

		vector<int> dp(size, 1);
		int maxlength = 0;
	
		for (int i=0;i<size;++i)
		{
			for (int j=0;j<i;++j)
			{
				if (nums[j]<nums[i])
				{
					dp[i] = max(dp[i],dp[j]+1);
				}
			}
			maxlength = max(maxlength, dp[i]);
		}
	
		return maxlength;
		
	}

};
```



### [最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

```c++
class Solution {
public:
	//最长公共子序列问题是典型的二维动态规划问题。
	int longestCommonSubsequence(string text1, string text2) {
		int m = text1.size();
		int n = text2.size();

		//dp[i][j]表示到字符s1[i],s2[j]位置的最大公共子序列
		vector<vector<int>> dp(m+1,vector<int>(n+1,0));
	
		for (int i=1;i<=m;++i)
		{
			for (int j=1;j<=n;++j)
			{
				if (text1[i-1] == text2[j-1])
				{//相同字符加1
					dp[i][j] = dp[i - 1][j - 1]+1;
				}
				else
				{
					dp[i][j] = max(dp[i - 1][j], dp[i][j-1]);
				}
			}
		}
	
		return dp[m][n];
	}

};
```



## 背包问题



### [分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```c++
class Solution {
public:
	bool canPartition(vector<int>& nums) {
		int n = nums.size();
		if (n < 2) {
			return false;
		}
		int sum = accumulate(nums.begin(), nums.end(), 0);
		int maxNum = *max_element(nums.begin(), nums.end());
		if (sum & 1) {
			return false;
		}
		int target = sum / 2;
		if (maxNum > target) {
			return false;
		}
		vector<vector<int>> dp(n, vector<int>(target + 1, 0));
		for (int i = 0; i < n; i++) {
			dp[i][0] = true;
		}
		dp[0][nums[0]] = true;
		

		for (int i = 1; i < n; i++) {
			int num = nums[i];
			for (int j = 1; j <= target; j++) {
				if (j >= num) {
					dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
				}
				else {
					dp[i][j] = dp[i - 1][j];
				}
			}
		}
		return dp[n - 1][target];
	}

};
```



## 吃掉 N 个橘子的最少天数

腾讯3月21日笔试题目

方法一：

时间复杂度超过要求

```c++
class Solution {
public:
  int GetMin(int a,int b,int c)
  {
​    if(c<=a &&c <=b)
​    {
​      return c;
​    }
​    if(a<=c&&a<=b)
​    {
​      return a;
​    }
​    return b;
  }
  int minDays(int n) {
​    int num =n;
​     //设置初始值，直接返回
​    if (num<=0)
​    {
​      return 0;
​    }
​    if (num == 1)
​    {
​      return 1;
​    }
​    if (num==2 || num == 3)
​    {
​      return 2;
​    }
​    //dp数组设置初始值
​    //dp数组 需要一个一个数值进行计算，空间比较浪费
​    vector<int> dp(num+1,0);
​    //1到0变换只需要1次，1-1=0
​    dp[1] = 1;
​    //2到0变换需要2次，2-1-2=0或则2/2-1=0
​    dp[2] = 2;
​    //3到0变换需要2次，3/3-1=0
​    dp[3] = 2;
​    for (int i=4;i<=num;++i)
​    {
​      int tmp1 = INT_MAX;
​      if (i%2==0)
​      {
​        tmp1 = dp[i-i / 2] ;
​      }
​      int tmp2 = INT_MAX;
​      if (i % 3 == 0)
​      {
​        tmp2 = dp[i-2*(i / 3)] ;
​      }
​      //dp[i]的值是三个变换的最小值+1
​      dp[i] = GetMin(dp[i-1], tmp1, tmp2)+1;
​    }
​    return dp[num];
  }
};
```



方法二

```c++
class Solution {

public:

  int minDays(int n) {
​    if(n<=1)
​    {
​      return 1;
​    }  
​    if(mii[n])
​    {
​      return mii[n];
​    }
​    return mii[n] = min(n%2+1+minDays(n/2),n%3+1+minDays(n/3));
  }

private:

  unordered_map<int,int> mii;

};
```



# 贪心法

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485087&idx=2&sn=e74bdfdae607939a169295f7f95eff7a&scene=21#wechat_redirect



## 区间调度

### 1 无重叠区间

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

https://leetcode-cn.com/problems/non-overlapping-intervals/



```c++
int eraseOverlapIntervals(vector<vector<int>>& intervals) 
{
	if (intervals.size()<=1)
	{
		return 0;
	}
	sort(intervals.begin(), intervals.end(), [](vector<int> &va, vector<int> &vb) 
	{
		return va[1] < vb[1];
	});

	vector<int> pre = intervals[0];
	int res = 0;
	for (int i=1;i< intervals.size();++i)
	{
		//开始区间<结束区间，有重叠
		if (intervals[i][0]<pre[1])
		{
			res++;
		}
		else
		{//没有重叠
			pre = intervals[i];
		}
	}
	
	return res;

}
```



### 2 用最少数量的箭引爆气球

https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/

```c++
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.size() <1)
	{
		return 0;
	}
	sort(points.begin(), points.end(), [](vector<int> &va, vector<int> &vb)
	{
		return va[1] < vb[1];
	});

	vector<int> pre = points[0];
	//初始为1
	int res = 1;
	for (int i = 1; i < points.size(); ++i)
	{
		//开始区间<结束区间，有重叠
		if (points[i][0] <= pre[1])
		{
		}
		else
		{//没有重叠，需要再用一个箭
			res++;
			pre = points[i];
		}
	}
	
	return res;
	}

};
```



## 分配问题

### 1 Assign Cookies (Easy)

题目描述

有一群孩子和一堆饼干，每个孩子有一个饥饿度，每个饼干都有一个大小。每个孩子只能吃 一个饼干，且只有饼干的大小不小于孩子的饥饿度时，这个孩子才能吃饱。求解最多有多少孩子 可以吃饱。

贪心策略：

给剩余孩子里最小饥饿度的孩子分配最小的能饱腹的饼干（先排序）

```c++
int findContentChildren(vector<int>& g, vector<int>& s) 
{
	sort(g.begin(),g.end());
	sort(s.begin(),s.end());
	int child = 0, cookie = 0;
	while(child<g.size() && cookie<s.size())
	{
		if (g[child]<=s[cookie])
		{
			child++;
		}
		cookie++;
	}
	return child;
}
```



### 2 Candy (Hard)

题目描述

一群孩子站成一排，每一个孩子有自己的评分。现在需要给这些孩子发糖果，规则是如果一 个孩子的评分比自己身旁的一个孩子要高，那么这个孩子就必须得到比身旁孩子更多的糖果;所 有孩子至少要有一个糖果。求解最少需要多少个糖果。

输入输出样例

```
输入是一个数组，表示孩子的评分。输出是最少糖果的数量。
Input: [1,0,2]
Output: 5
```

在这个样例中，最少的糖果分法是 [2,1,2]。

贪心策略：

在每次遍历中，只考虑并更新相邻一侧的大小关系

```c++
int candy(vector<int>& ratings) {
        //有相邻的要求，所以不能先排序
		int childsize = ratings.size();
		//每个孩子至少分配到 1 个糖果
		vector<int> cookies(childsize,1);
		//从左向右遍历，如果右边的孩子分数大于左边孩子的分数，则右边孩子的糖果数=左边孩子的糖果数+1，否则为1
		for (size_t i = 0; i < childsize; i++)
		{
			if (i>0 && ratings[i]>ratings[i-1])
			{
				/* code */
				cookies[i] = cookies[i-1]+1;
			}
		}

		//从右向左遍历，如果左边的孩子分数大于右边孩子的分数，则左边孩子的糖果数=右边孩子的糖果数+1，否则为1
		for (size_t i = childsize-1; i>0; i--)
		{
			/* code */
			if(ratings[i]<ratings[i-1])
			{
				cookies[i-1] = max(cookies[i]+1,cookies[i-1]);
			}
		}
	
		int res = accumulate(cookies.begin(),cookies.end(),0);
		return res;
		
	}
```



# 图

# LRU

注：

1 使用双向链表
2 超出时间限制可把打印语句注释掉
3 双向链表中需要保存map的key和value，因为删除节点时也需要同步删除map中的元素，需要知道key的值
4 需要定义链表length

```c++
struct DLinkNode
{
    int m_value;
    int m_key;//需要定义key，删除节点时，也需要删除map中元素，是通过key进行删除的
    DLinkNode *prev;
    DLinkNode *next;//使用双指针是可以在O（1）的时机复杂度完成插入和删除操作
    DLinkNode(int key=0,int value=0):m_key(key),m_value(value),prev(nullptr),next(nullptr){}
};

class LRUCache {
public:
    LRUCache(int capacity) {
        //初始化构造节点
        m_capacity = capacity;
        m_head = new DLinkNode();
        m_tail = new DLinkNode();
        m_head->prev = nullptr;
        m_tail->next = nullptr;
        m_head->next = m_tail;
        m_tail->prev = m_head;
        size = 0;
    }
    

    int get(int key) {
        //std::cout<<"get:"<<key<<std::endl;
        if(key2node.count(key))
        {//节点存在
            // 1 找到该节点的值
            // 2 调整该节点在链表的位置，移动节点到链表最前面（不需要直接移动，可先删除，然后添加，调整指针即可）
            DLinkNode *node = key2node.at(key);
            int value = node->m_value;
            DeleteNode(node);
            AddNodeToHead(node);
            return value;
        }
        else
        {//节点不存在
            return -1;
        }
        
    }
    
    void put(int key, int value) {
        //std::cout<<"put:"<<key<<std::endl;
        if(key2node.count(key))
        {//节点存在,更新节点值即可
            DLinkNode *node = key2node.at(key);
            node->m_value = value;
            DeleteNode(node);
            AddNodeToHead(node);
        }
        else
        {//节点不存在
            //1 构造新节点
            DLinkNode *newnode =  new DLinkNode(key,value);
            key2node[key] = newnode;
            ++size;
            //2 判断链表长度是否已经是最大长度
            if(size>m_capacity)
            {//3 如果是最大长度，删除链表最后的节点，添加新节点到头部
                //最后一个节点，是最久未使用的
                DLinkNode *lastnode =DeleteLastNode();
                AddNodeToHead(newnode);
                key2node.erase(lastnode->m_key);
                delete lastnode;
                lastnode = nullptr;
            }
            else
            {//4 如果不是最大长度，添加新节点到头部
                AddNodeToHead(newnode);
            }
        }
    }
    
    void AddNodeToHead(DLinkNode *node)
    {
        //std::cout<<"AddNodeToHead begin"<<std::endl;
        m_head->next->prev = node;
        node->prev = m_head;
        node->next = m_head->next;
        m_head->next = node;
        //std::cout<<"AddNodeToHead end"<<std::endl;
    }
    
    DLinkNode *DeleteLastNode()
    {
        DLinkNode *lastnode = m_tail->prev;
        //m_tail->prev->next = m_tail;
        //m_tail->prev = m_tail->prev->prev;
        //上面注释的代码是错误，代码能复用就尽量复用，m_tail->prev还是要删除的节点
        DeleteNode(lastnode);
        return lastnode;
    }
    
    void DeleteNode(DLinkNode *node)
    {
        //std::cout<<"DeleteNode0"<<std::endl;
        node->prev->next = node->next;
        node->next->prev= node->prev;
        //std::cout<<"DeleteNode2"<<std::endl;
    }

private:
    int m_capacity = -1;
    unordered_map<int,DLinkNode*> key2node;
    DLinkNode *m_head;
    DLinkNode *m_tail;
    int size = -1;  //链表长度，需要定义当前链表长度是否超过最大长度
};
```



# 一切皆可搜索

深度优先搜索使用栈模拟

广度优先搜索使用队列模拟

## DFS

### [岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```c++
class Solution {
public:
	int maxAreaOfIsland(vector<vector<int>>& grid) {
		int max_area = 0;
		int m = grid.size();
		int n = grid[0].size();
		for (int i=0;i<m;++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (grid[i][j])
				{
					max_area = max(max_area,dfs(i,j,grid));
				}
			}
		}

        return max_area;
    }
    
    int dfs(int row,int col, vector<vector<int>>& grid)
    {
    	if (row<0 || row>= grid.size() || col<0 || col>= grid[0].size() || grid[row][col]==0)
    	{
    		return 0;
    	}
    	grid[row][col] = 0;
    	return 1 + dfs(row+1,col,grid)+ dfs(row - 1, col, grid)+
    		dfs(row, col+1, grid)+ dfs(row, col-1, grid);
    }

};
```





### [省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

（连通分量）

有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

```c++
class Solution {
public:
	//主函数
	int findCircleNum(vector<vector<int>>& isConnected) {
		int res = 0;
		vector<bool> visied(isConnected.size(),false);
		//每一行看作一个节点
		for (int i=0;i< isConnected.size();++i)
		{
				if (!visied[i])
				{
					dfs(i, visied,isConnected);
					res++;
				}
		}

		return res;
	}
	//辅函数
	void dfs(int row, vector<bool> &visied,vector<vector<int>>& isConnected)
	{
		visied[row] = true;
		for (int i=0;i< isConnected.size();++i)
		{
			if (isConnected[row][i] && !visied[i])
			{
				dfs(i,visied,isConnected);
			}
		}
	}

};
```

### [太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

```c++
class Solution {
public:
	vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
		int m = heights.size();
		int n = heights[0].size();
		vector<vector<bool>> vvb_first(m,vector<bool>(n,false));
		vector<vector<bool>> vvb_second(m,vector<bool>(n,false));
		vector<vector<int>> res;
		for (int col=0;col<n;++col)
		{
			dfs(0,col,heights, heights[0][col],vvb_first);
			dfs(m-1,col,heights, heights[m-1][col], vvb_second);
		}

		for (int row = 0; row < m; ++row)
		{
			dfs(row, 0, heights, heights[row][0], vvb_first);
			dfs(row, n-1, heights, heights[row][n-1], vvb_second);
		}
	
		for (int i=0;i<m;++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (vvb_first[i][j] && vvb_second[i][j])
				{
					res.push_back({i,j});
				}
			}
		}
	
		return res;
	}
	
	void dfs(int row,int col, vector<vector<int>>& heights,int &value, vector<vector<bool>> &vvb)
	{
		if (row<0 || row>= heights.size() || col<0 || col>= heights[0].size() || vvb[row][col])
		{
			return;
		}
	
	    if(heights[row][col]<value)
	    {//必须提前返回，否则会一直向下遍历
	        return;
	    }
	
		if (heights[row][col]>=value)
		{
			vvb[row][col] = true;
		}
	
		dfs(row+1,col,heights, heights[row][col],vvb);
		dfs(row-1,col,heights, heights[row][col],vvb);
		dfs(row,col+1,heights, heights[row][col],vvb);
		dfs(row,col-1,heights, heights[row][col],vvb);
	}

};
```



## BFS

**BFS 找到的路径一定是最短的**

**问题的本质就是让你在一幅「图」中找到从起点`start`到终点`target`的最近距离**

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485134&idx=1&sn=fd345f8a93dc4444bcc65c57bb46fc35&scene=21#wechat_redirect

```java
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路

    q.offer(start); // 将起点加入队列
    visited.add(start);
    int step = 0; // 记录扩散的步数
    
    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 划重点：这里判断是否到达终点 */
            if (cur is target)
                return step;
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj())
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
        }
        /* 划重点：更新步数在这里 */
        step++;
    }

}
```

### [ 打开转盘锁](https://leetcode-cn.com/problems/open-the-lock/)

```c++
class Solution {
public:

	string UpLock(string str,int index)
	{
		string tmp = str;
		if (str[index] == '9')
		{
			tmp[index] = '0';
		}
		else
		{
			tmp[index]+=1;
		}
	
		return tmp;
	}
	
	string DownLock(string str, int index)
	{
		string tmp = str;
		if (str[index] == '0')
		{
			tmp[index] = '9';
		}
		else
		{
			tmp[index] -= 1;
		}
	
		return tmp;
	}
	
	int openLock(vector<string>& deadends, string target) {
		unordered_set<string> deadsets;
		for (auto deadstr: deadends)
		{
			deadsets.insert(deadstr);
		}
	
		// 记录已经穷举过的密码，防止走回头路
		unordered_set<string> visited;
	
		queue<string> qs;
		qs.push("0000");
		visited.insert("0000");
		int steps = 0;
		while (!qs.empty())
		{
			int size = qs.size();
			for (int i=0;i< size;++i)
			{
				string tmp = qs.front();
				qs.pop();
	
				if (deadsets.count(tmp))
				{
					continue;
				}
	
				if (tmp == target)
				{
					return steps;
				}
				for (int j=0;j<4;++j)
				{
					string upstring = UpLock(tmp,j);
					if (!visited.count(upstring))
					{
						qs.push(upstring);
						visited.insert(upstring);
					}
					string downstring = DownLock(tmp,j);
					if (!visited.count(downstring))
					{
						qs.push(downstring);
						visited.insert(downstring);
					}
				}
				
			}
			steps++;
		}
	
		return -1;
	}

};
```

### [二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

```c++
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root)
		{
			return 0;
		}

		int steps = 1;
		queue<TreeNode *> qnodes;
		qnodes.push(root);
		while (!qnodes.empty())
		{
			int size = qnodes.size();
			for (int i = 0; i < size;++i)
			{
				TreeNode *tmp = qnodes.front();
				qnodes.pop();
				if (!tmp->left  && !tmp->right)
				{
					return steps;
				}
				if (tmp->left)
				{
					qnodes.push(tmp->left);
				}
				if (tmp->right)
				{
					qnodes.push(tmp->right);
				}
			}
	
			steps++;
		}
	
		return 0;
	}

};
```

### [最短的桥](https://leetcode-cn.com/problems/shortest-bridge/)

在给定的二维二进制数组 A 中，存在两座岛。（岛是由四面相连的 1 形成的一个最大组。）

现在，我们可以将 0 变为 1，以使两座岛连接起来，变成一座岛。

返回必须翻转的 0 的最小数目。（可以保证答案至少是 1 。）

```c++
class Solution {
public:
	vector<int> direction{-1,0,1,0,-1};

	int shortestBridge(vector<vector<int>>& grid) {
		int m = grid.size();
		int n = grid[0].size();
		queue<pair<int, int>> points;
		bool flipped = false;
		for (int i=0;i<m;++i)
		{
			if (flipped)
			{
				break;
			}
			for (int j = 0;j<n;++j)
			{
				if (grid[i][j])
				{//从grid[i][j]=1的位置开始扩散，找到一个岛屿
					//使用深度优先遍历找到这个岛屿的所有位置，并将其位置置为2
					dfs(points,grid,m,n,i,j);
					flipped = true;
					break;
				}
			}
		}
	
		//bfs搜索最短路径
		int level = 0;
		while (!points.empty())
		{
			++level;
			int size = points.size();
			for (int i=0;i<size;++i)
			{
				auto[r, c] = points.front();
				points.pop();
	
				for (int k =0;k<4;++k)
				{
					int x = r + direction[k];
					int y = c + direction[k+1];
					if (x>=0 && y>=0 && x<m && y<n)
					{//在区间中
						if (grid[x][y] == 2)
						{
							continue;
						}
						else if (grid[x][y] == 1)
						{
							return level;
						}
	
						points.push({x,y});
						grid[x][y] = 2;
					}
				}
			}
		}
	
		return 0;
	}
	
	void dfs(queue<pair<int, int>> &points,vector<vector<int>>& grid,int m,int n,int i,int j)
	{
		//不要超过边界，grid[i][j] == 2说明该位置已经进行过搜索
		if (i<0 || j<0 || i>=m || j>=n || grid[i][j] == 2)
		{
			return;
		}
	
		if (grid[i][j] == 0)
		{
			points.push({i,j});
			return;
		}
	
		grid[i][j] = 2;
		dfs(points,grid,m,n,i-1,j);
		dfs(points,grid,m,n,i+1,j);
		dfs(points,grid,m,n,i,j-1);
		dfs(points,grid,m,n,i,j+1);
	}

};
```

### [单词接龙](https://leetcode-cn.com/problems/word-ladder/)

字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：

序列中第一个单词是 beginWord 。
序列中最后一个单词是 endWord 。
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典 wordList 中的单词。
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。

```c++
class Solution {
public:
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> wordsets;
		for (auto word:wordList)
		{
			wordsets.insert(word);
		}

        wordsets.erase(beginWord);
    	unordered_set<string> visited;
    	queue<string> qs;
    	qs.push(beginWord);
    	visited.insert(beginWord);
    
    	int step = 1;
    
    	while (!qs.empty())
    	{
    		++step;
    		int size = qs.size();
    		for (int i=0;i<size;++i)
    		{
    			string tmp = qs.front();
    			qs.pop();


				//改变单词的每一个字符
				int wordlen = tmp.size();
				for (int w = 0;w<wordlen;++w)
				{
					char orionch = tmp[w];
					for (char j='a';j<='z';j++)
					{
						if (j == orionch)
						{
							continue;
						}
	
						tmp[w] = j;
						if (wordsets.count(tmp))
						{
							if (tmp == endWord)
							{
								return step;
							}
	
							if (!visited.count(tmp))
							{
								qs.push(tmp);
								visited.insert(tmp);
							}
						}
	
					}
	
					tmp[w] = orionch;
				}
			}
		}
	
		return 0;
	}

};
```



### [单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

```c++
class Solution {
public:
	vector<vector<string>> res;

	vector<vector<string>>  findLadders(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> wordsets;
		for (auto word:wordList)
		{
			wordsets.insert(word);
		}
	
		wordsets.erase(beginWord);
	
		unordered_map<string, unordered_set<string>> from = { {beginWord, {}} };
	
		unordered_map<string, int> steps = { {beginWord, 0} };
	
		queue<string> qs;
		qs.push(beginWord);
	
		int step = 1;
		bool find = false;
	
		while (!qs.empty())
		{
			++step;
			int size = qs.size();
			for (int i=0;i<size;++i)
			{
				string tmp = qs.front();
				qs.pop();


				//改变单词的每一个字符
				int wordlen = tmp.size();
				string currentword = tmp;
				for (int w = 0;w<wordlen;++w)
				{
					char orionch = tmp[w];
					for (char j='a';j<='z';j++)
					{
						if (j == orionch)
						{
							continue;
						}
	
						tmp[w] = j;
	
	                    if (steps[tmp] == step)
						{//可能存在另一条路径到达tmp,tmp之前已经访问过，如果step相同，直接加入from即可（step相同说明处在最小层数）
	                        std::cout<<"steps[tmp]:"<<steps[tmp] <<std::endl;
	                        std::cout<<"step:"<<step <<std::endl;
	                        std::cout<<"tmp:"<<tmp <<std::endl;
	                        from[tmp].insert(currentword);
						}


                        // 如果从一个单词扩展出来的单词以前遍历过，距离一定更远，
                        //为了避免搜索到已经遍历到，且距离更远的单词，需要将它从 dict 中删除
                        if(!wordsets.count(tmp))
                        {
                            continue;
                        }
    
                        wordsets.erase(tmp); 
                        qs.push(tmp);
                        from[tmp].insert(currentword);
    
                        if (tmp == endWord)
    					{
    						//找到之后不要直接退出，这一层的其他节点也有可能转换到endWord
    						find = true;
    					}
                        steps[tmp] = step;
    
    				}
    
    				tmp[w] = orionch;
    			}
    		}
    
    		if (find)
    		{
    			break;
    		}
    	}
    
    	//使用回溯法找出所有的可行解
    	if (find) {
    		vector<string> Path = { endWord };
    		dfs(res, endWord, from, Path);
    	}
    	return res;
    }
    
    void dfs(vector<vector<string>> &res,string node,
    	unordered_map<string,unordered_set<string>> &from, vector<string>&path)
    {
    	if (from[node].empty())
    	{
    		res.push_back({ path .rbegin(),path.rend()});
    		return;
    	}
    	for (auto setstrs: from[node])
    	{
    		path.push_back(setstrs);
    		dfs(res, setstrs,from,path);
    		path.pop_back();
    	}
    }

};
```



## 回溯法

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484709&idx=1&sn=1c24a5c41a5a255000532e83f38f2ce4&scene=21#wechat_redirect

```c++
vector<vector<string>> res;

/* 输入棋盘边长 n，返回所有合法的放置 */
vector<vector<string>> solveNQueens(int n) {
    // '.' 表示空，'Q' 表示皇后，初始化空棋盘。
    vector<string> board(n, string(n, '.'));
    backtrack(board, 0);
    return res;
}

// 路径：board 中小于 row 的那些行都已经成功放置了皇后
// 选择列表：第 row 行的所有列都是放置皇后的选择
// 结束条件：row 超过 board 的最后一行
void backtrack(vector<string>& board, int row) {
    // 触发结束条件
    if (row == board.size()) {
        res.push_back(board);
        return;
    }

    int n = board[row].size();
    for (int col = 0; col < n; col++) {
        // 排除不合法选择
        if (!isValid(board, row, col)) 
            continue;
        // 做选择
        board[row][col] = 'Q';
        // 进入下一行决策
        backtrack(board, row + 1);
        // 撤销选择
        board[row][col] = '.';
    }
}
```

### [全排列](https://leetcode-cn.com/problems/permutations/)

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

```
class Solution {
public:
	vector<vector<int>> permute(vector<int>& nums) {
		vector<int> track;
		dfs(nums,track);
		return res;
	}

	//track已经选择的路径
	void dfs(vector<int>& nums, vector<int>& track)
	{
		if (track.size() == nums.size())
		{
			res.push_back(track);
			return;
		}
	
		for (int i= 0;i< nums.size();++i)
		{
			auto it = std::find(track.begin(), track.end(),nums[i]);
			if (it!=track.end())
			{//排除重复数字
				continue;
			}
			track.push_back(nums[i]);
			dfs(nums,track);
			track.pop_back();
		}
	}
	
	vector<vector<int>> res;

};
```

### [组合](https://leetcode-cn.com/problems/combinations/)

给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案。

```c++
class Solution {
public:
	vector<vector<int>> combine(int n, int k) {
		vector<int> track;
		dfs(n,track,0,k);
        return res;
	}

	void dfs(int n, vector<int>& track,int index,int k)
	{
		if (track.size() == k)
		{
			res.push_back(track);
			return;
		}
	
		for (int i=index;i<n;++i)
		{
			int num = i + 1;
			track.push_back(num);
			dfs(n,track,i+1,k);
			track.pop_back();
		}
	}
	
	vector<vector<int>> res;

};
```



### [N 皇后](https://leetcode-cn.com/problems/n-queens/)

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

```c++
class Solution {
public:
	vector<vector<string>> solveNQueens(int n) {

		vector<string> track(n,string(n,'.'));
		dfs(track,0,n);
		return res;
	}
	
	void dfs(vector<string> &track,int row,int n)
	{
		if (row == n)
		{//没有选择列表
			res.push_back(track);
			return;
		}
	
		for (int col=0;col<n;++col)
		{
			if (!IsValid(track,row,col))
			{
				continue;
			}
	
			track[row][col] = 'Q';
			dfs(track,row+1,n);
			track[row][col] = '.';
		}
	}
	
	//track[row][col]是否合理
	bool IsValid(vector<string> &track,int row,int col)
	{
		//不需要检查行，每一行放置一个皇后，不会重复
		//检查列是否合理
		int n = track.size();
		for (int i=0;i<n;++i)
		{
			if (track[i][col]=='Q')
			{//先排除不合理的选择，track[row][col]此时还没有防止皇后
				return false;
			}
		}
	
		//检查右上方是否合理，右下方还没有进行放置皇后
		for (int i=row,j = col;i>=0 && j<=n-1;i--,j++)
		{
			if (track[i][j] == 'Q')
			{//先排除不合理的选择，track[row][col]此时还没有防止皇后
				return false;
			}
		}
	
		//检查左上方是否合理，左下方还没有防止皇后
		for (int i = row, j = col; i >= 0 && j>=0; i--, j--)
		{
			if (track[i][j] == 'Q')
			{//先排除不合理的选择，track[row][col]此时还没有防止皇后
				return false;
			}
		}
		return true;
	}
	
	vector<vector<string>> res;

};
```



### [ 单词搜索](https://leetcode-cn.com/problems/word-search/)

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```c++
class Solution {
public:
	bool find = false;
	bool exist(vector<vector<char>>& board, string word) {
		int m = board.size();
		int n = board[0].size();
		vector<vector<bool>> visited(m,vector<bool>(n,false));
		for (int i=0;i<m;++i)
		{
			for (int j=0;j<n ;++j)
			{
				dfs(i,j,board,visited,word,0);
			}
		}

		return find;
	}
	
	void dfs(int row, int col, vector<vector<char>>& board, vector<vector<bool>>& visited, string word, int index)
	{
		if (row<0 || row>= board.size() || col<0|| col>= board[0].size())
		{
			return;
		}
		if (visited[row][col] || find || board[row][col]!=word[index])
		{
			return;
		}
	
		if (index == word.size()-1)
		{
			find = true;
			return;
		}
	
		visited[row][col] = true;
	
		dfs(row+1,col,board,visited,word,index+1);
		dfs(row-1,col,board,visited,word,index+1);
		dfs(row,col+1,board,visited,word,index+1);
		dfs(row,col-1,board,visited,word,index+1);
	
		visited[row][col] = false;
	}

};
```

