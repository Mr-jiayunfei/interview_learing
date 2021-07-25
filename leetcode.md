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

# 滑动窗口

## 框架总结

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485141&idx=1&sn=0e4583ad935e76e9a3f6793792e60734&scene=21#wechat_redirect

```c++
int left = 0, right = 0;

while (right < s.size()) {
    // 增大窗口
    window.add(s[right]);
    right++;

    while (window needs shrink) {
        // 缩小窗口
        window.remove(s[left]);
        left++;
    }

}
```

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



## 长度最小的子数组

https://leetcode-cn.com/problems/minimum-size-subarray-sum/solution/chang-du-zui-xiao-de-zi-shu-zu-by-leetcode-solutio/

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 



### 方法1 暴力法：

```C++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        if(nums.empty() || target <0)
        {
         return -1;
        }
  int ret = 0;

  int minlength = INT32_MAX;

  for(int i=0;i<nums.size();++i)
  {
    int indexsum = nums[i];
    int length=1;
    if(indexsum>=target)
    {//有可能只有一个数字就和目标数字相同或者大于目标数字
        return 1;
    }
    for(int j=i+1;j<nums.size();++j)
    {
      indexsum+=nums[j];
      length++;
      if(indexsum>=target)
      {
        if(minlength>length)
        {
          minlength =length;
        }
        break;
      }
    }
  }
  if(minlength == INT32_MAX)
  {
    ret = 0;
  }
  else
  {
    ret = minlength;
  }

  return ret;
    }
};
```



### 暴力法优化，代码不那么冗余

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j < n; j++) {
                sum += nums[j];
                if (sum >= s) {
                    ans = min(ans, j - i + 1);
                    break;
                }
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
```



### 方法2 滑动窗口法

时间复杂度为o(n),空间复杂度为o(1)

```c++
class Solution {
public:


    int GetSum(const vector<int>  &vi,int left,int right)
    {
        int sum = 0;
        for(int i=left;i<=right && i<vi.size();++i)
        {
            sum+= vi[i];
        }
        return sum;
    }
    
    int minSubArrayLen(int target, vector<int>& nums) {
        if(nums.empty() || target <0)
        {
            return -1;
         }
      
        int ret = 0;
        int left = 0,right = 0;
        int numssize = nums.size();
        int minlength = INT32_MAX;
        while(right<numssize) 
        {
            int tempsum  = GetSum(nums,left,right);
            if(tempsum<target)
            {
           		 right++;
            }
            else
            {
           	 	if(minlength>right-left+1)
           		 {
               	 minlength = right-left+1;
           		 }
           		 left++;
            }
        }
        
        if(minlength == INT32_MAX)
        {
            ret = 0;
        }
        else
        {
            ret = minlength;
        }
        
        return ret;
    }

};
```



### 滑动窗口优化，代码不那个冗余

时间复杂度为o(n),空间复杂度为o(1)

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
```





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

大顶推

小顶推

## 快速排序

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



# 查找

## 二分查找



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

# 双指针

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

## **455. Assign Cookies (Easy)**

题目描述

有一群孩子和一堆饼干，每个孩子有一个饥饿度，每个饼干都有一个大小。每个孩子只能吃 一个饼干，且只有饼干的大小不小于孩子的饥饿度时，这个孩子才能吃饱。求解最多有多少孩子 可以吃饱。



## 135. Candy (Hard)

题目描述

一群孩子站成一排，每一个孩子有自己的评分。现在需要给这些孩子发糖果，规则是如果一 个孩子的评分比自己身旁的一个孩子要高，那么这个孩子就必须得到比身旁孩子更多的糖果;所 有孩子至少要有一个糖果。求解最少需要多少个糖果。

输入输出样例

```
输入是一个数组，表示孩子的评分。输出是最少糖果的数量。
Input: [1,0,2]
Output: 5
```

在这个样例中，最少的糖果分法是 [2,1,2]。



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

