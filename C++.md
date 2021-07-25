# C++问题

## 1 为什么要有初始化表？

1 如果没有在构造函数的初始值列表中显示地初始化成员，则该成员在构造函数体之前执行默认初始化（类类型的参数执行默认构造函数，普通类型的参数值为随机数），即先定义然后在构造函数体内赋值。

2 如果成员时const或者引用的话，必须将其初始化，因为引用和const变量必须初始化，不能先定义后赋值

3 当成员属于某种类类型且该类没有定义默认构造函数时（提供的构造函数均为有参构造函数），必须将这个成员初始化，因为无法对该成员的构造函数进行调用

注：如果一个构造函数为所有参数提供了默认实参，则它实际上也定义了默认构造函数（提供了默认实参后不需要提供无参构造函数，否则会编译错误）

4 成员变量的初始化顺序由声明顺序决定，而与初始化表的顺序无关，

最好令构造函数初始值的顺序与成员声明的顺序一致，

最好令构造函数的参数作为成员的初始值，

尽量避免使用其他成员初始化其他成员。

5 对象的创建过程：

（1）为对象分配内存

（2）调用成员子对象的构造函数(声明顺序)

（3）执行构造函数的代码



## 2 基类的析构函数为什么要设计为虚函数？

基类中的析构函数如果是虚函数，那么派生类的析构函数就重写了基类的析构函数。

这里他们的函数名不相同，看起来违背了重写的规则，但实际上编译器对析构函数的名称做了特殊处理，编译后析构函数的名称统一处理成destructor。

子类的析构函数，会自动调用基类的析构函数，析构基类子对象。

注：析构函数的运作方式是最深层派生的那个子类其析构函数最先被调用，然后是其每一个基类的析构函数被调用，编译器会在子类的析构函数中创建一个对基类的析构函数的调用动作，所以基类的析构函数必须有定义。

基类的析构函数不会调用子类的析构函数，对一个指向子类对象的基类指针使用delete运算符，实际被执行的只有基类的析构函数，所释放的仅仅是基类子对象的动态资源，如果子类中有其它的动态资源分配将会形成内存泄露。

将基类的析构函数设计为虚函数，可利用多态调用子类的析构函数，进而调用基类的析构函数，释放所有资源。

注：必须是指针或者引用才有多态特性





## 3  智能指针有几种（shared_ptr,unique_ptr,weakPtr,autoptr）？每种智能指针的实现原理？每种智能指针的适用场景？为什么要有weakPtr？

智能指针是一个模板类，当离开作用域时，可以调用其析构函数自动释放内存资源。

智能指针有三种，shared_ptr，unique_ptr,weak_ptr。

shared_ptr允许多个指针指向同一个对象，unique_ptr则独占所指向的对象，weak_ptr是一种弱引用，指向shared_ptr所管理的对象。

shared_ptr的实现原理是在其内部维护了一个计数器（也是动态分配的内存），当我们拷贝一个shared_ptr时，计数器会递增，当我们给shared_ptr赋予一个新值或是shared_ptr被销毁时，计数器会递减。

unique_ptr的实现原理是拷贝构造和赋值运算符都被删除（或者定义为私有），保留了移动构造和移动赋值函数。

weak_ptr的实现原理是在其内部维护了一个weak_count计数器。

autoptr是C++11之前实现的智能指针，现在已经基本弃用，有两个明显缺陷：第一个是使用delete操作符释放资源，如果是动态分配的数组对象，无法进行资源释放，第二个是两个指针不能指向相同的对象，当进行赋值操作时，会将另一个指针置为空值。

shared_ptr和weak_ptr拥有相同的基类，在这个基类中维护了引用计数基类_Ref_count_base，这个基类中有两个数据成员，_Uses和_Weaks，

```c++
_Atomic_counter_t _Uses;
_Atomic_counter_t _Weaks;
```

_Uses则是我们理解的引用计数，每有一个shared_ptr共享资源，_Uses就会加一，反之每一个shared_ptr析构，_Uses就会减一，当_Uses变为零时，意味着没有shared_ptr再占有资源，这个时候占有的资源会调用释放操作。但是并不能直接销毁引用计数对象，因为可能有弱引用还绑定到引用计数对象上。

而_Weaks就是weak_ptr追踪资源的计数器，每有一个weak_ptr追踪资源，_Weaks就会加一，反之每一个weak_ptr析构时，_Weaks就会减一，当_Weaks变为零时，意味着没有weak_ptr再追踪资源，这时会销毁引用计数对象。（可使用VS21017查看智能指针的实现源码）

三种智能指针使用C++的简单实现代码参考如下：

https://github.com/Mr-jiayunfei/code_learn/blob/main/STL/implement_SmartPtr/main.cpp



weak_ptr的使用场景

weak_ptr只能从shared_ptr对象构建。

weak_ptr并不影响动态对象的生命周期，即其存在与否并不影响对象的引用计数器。当weak_ptr所指向的对象因为shared_ptr计数器为0而被释放后，那么weak_ptr的lock方法将返回空。

weak_ptr并没有重载operator->和operator *操作符，因此不可直接通过weak_ptr使用对象。

提供了expired()与lock()成员函数，前者用于判断weak_ptr指向的对象是否已被销毁，后者返回其所指对象的shared_ptr智能指针(对象销毁时返回”空“shared_ptr)，如果返回shared_ptr，那么计数器会加1.

`std::weak_ptr`是解决悬空指针问题的一种很好的方法。仅通过使用原始指针，不知道所引用的数据是否已被释放。

相反，通过`std::shared_ptr`管理数据并提供`std::weak_ptr`给用户，用户可以通过调用`expired()`或来检查数据的有效性`lock()`。

weak_ptr可以解决shared_ptr的循环引用问题

循环引用问题解释（

https://blog.csdn.net/leichaowen/article/details/53064294

https://zhuanlan.zhihu.com/p/355812360

）



## 4 Shared_ptr的C++简单实现

```c++
/*
	SharedPtr智能指针的简单实现
*/

template<typename T>
class SamrtPtr
{
public:
	//构造函数
	SamrtPtr(T* ptr = nullptr) :m_ptr(ptr) {
		if (m_ptr) {
			m_count = new size_t(1);
		}
		else {
			m_count = new size_t(0);
		}
	}

	//拷贝构造函数
	SamrtPtr(const SamrtPtr&that)
	{
		if (this != &that) {
			this->m_ptr = that.m_ptr;
			//指针指向的实际上是同一个计数器，同一块内存
			this->m_count = that.m_count;
			(*this->m_count)++;
		}
	}
	
	//拷贝赋值函数
	SamrtPtr&operator=(const SamrtPtr&that)
	{
		//指针指向的实际上是同一个计数器，同一块内存
		if (this->m_ptr == that.m_ptr)
		{
			return *this;
		}
	
		if (this->m_ptr)
		{
			//将原来的引用计数减一
			(*this->_count)--;
			if (this->_count == 0) {
				delete this->_ptr;
				delete this->_count;
			}
		}
	
		this->m_ptr = that.m_ptr;
		this->m_count = that.m_count;
		(*this->m_count)++;
		return *this;
	}
	
	//重载*操作符
	T& operator*()
	{
		return *(this->m_ptr);
	}
	
	T *operator->()
	{
		return this->m_ptr;
	}
	
	~SamrtPtr()
	{
		(*this->m_count)--;
		if (*this->m_count == 0)
		{
			delete this->m_ptr;
			delete this->m_count;
		}
	}
	
	size_t use_count() {
		return *this->m_count;
	}

private:
	//指针
	T* m_ptr;
	//计数器
	size_t* m_count;
};
```



## 5 Unique_ptr的简单实现

```c++
/*
UniquePtr的简单实现
*/

template<typename T>
class UP
{
	T*   data;
public:
	// Explicit constructor
	explicit UP(T* data)
		: data(data)
	{}
	~UP()
	{
		delete data;
	}
	// Remove compiler generated methods.
	UP(UP const&) = delete;
	UP& operator=(UP const&) = delete;

	// Const correct access owned object
	T* operator->() const { return data; }
	T& operator*()  const { return *data; }
	
	// Access to smart pointer state
	T* get()                 const { return data; }
    /*
    if (obj)
    This will call the operator bool(), return the result, and use the result as the condition of the if
    */
	explicit operator bool() const { return data; }
	
	// Modify object state
	T* release()
	{
		T* result = nullptr;
		std::swap(result, data);
		return result;
	}

};
```



## 6 WeakPtr的简单实现

```c++
/*===========================================================================*/
/*
WeakPtr的简单实现
*/
/*
Counter对象的目地就是用来申请一个块内存来存引用计数和弱引用计数。shareCount是SharedPtr的引用计数，weakCount是弱引用计数。
当shareCount为0时，删除T*对象。
当weakCount为0同时shareCount为0时，删除Counter*对象。
*/
class Counter
{
public:
	int shareCount = 0;
	int weakCount = 0;
};

/*
SharedPtr类
主要的成员函数包括：
默认构造函数
参数为T*的explicit单参数构造函数
参数为WeakPtr&的explicit单参数构造函数
拷贝构造函数
拷贝赋值函数
析构函数
隐式类型转换操作符 operator bool ()
operator -> ()
operator * ()
reset()
get()
use_count()
*/

template<class T> class WeakPtr;
template<class T> class SharedPtr
{
public:
	friend class WeakPtr<T>; //方便weak_ptr与share_ptr设置引用计数和赋值。

	SharedPtr()
		: m_pResource(nullptr)
		, m_pCounter(nullptr)
	{
	}
	
	explicit SharedPtr(T* pResource = nullptr) // 禁止隐式装换
		: m_pResource(pResource)
		, m_pCounter(nullptr)
	{
		if (m_pResource != nullptr)
		{
			m_pCounter = new Counter;
			m_pCounter->shareCount = 1;
		}
	}
	
	SharedPtr(const WeakPtr<T>& other) // 供WeakPtr的lock()使用
		: m_pResource(other.m_pResource)
		, m_pCounter(other.m_pCounter)
	{
		if (m_pCounter != nullptr && 0 == m_pCounter->shareCount)
		{
			m_pResource = nullptr;
		}
	}
	
	SharedPtr(const SharedPtr<T>& other)
		: m_pResource(other->m_pResource)
		, m_pCounter(other->m_pCounter)
	{
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->shareCount); // 增加引用计数
		}
	}
	
	SharedPtr<T>& operator = (const SharedPtr<T>& other)
	{
		if (this == &other) return *this;
	
		release();
		m_pCounter = other.m_pCounter;
		m_pResource = other.m_pResource;
	
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->shareCount); // 增加引用计数
		}
	
		return *this;
	}
	
	~SharedPtr()
	{
		release();
	}
	
	operator bool()
	{
		return m_pResource != nullptr;
	}
	
	T& operator * ()
	{
		// 如果nullptr == m_pResource，抛出异常
		return *m_pResource;
	}
	
	T* operator -> ()
	{
		return m_pResource;
	}
	
	void Reset(T* pOther = nullptr)
	{
		release();
	
		m_pResourse = pOther;
		if (m_pResourse != nullptr)
		{
			m_pCounter = new Counter();
			m_pCounter->shareCount = 1;
		}
	}
	
	T* get()
	{
		return m_pResource;
	}
	
	int use_count()
	{
		return (m_pCounter != nullptr) ? m_pCounter->shareCount : 0;
	}

private:
	void release()
	{
		if (nullptr == m_pCounter) return;

		// T*肯定由SharedPtr释放，Counter*如果没有WeakPtr，也由SharedPtr释放
		--m_pCounter->shareCount;
	
		if (0 == m_pCounter->shareCount)
		{
			delete m_pResource;
			m_pResource = nullptr;
	
			if (0 == m_pCounter->weakCount)
			{
				delete m_pCounter;
				m_pCounter = NULL;
			}
		}
	}

public:
	T* m_pResource = nullptr;
	Counter* m_pCounter = nullptr;
};

/*
WeakPtr类
主要的成员函数包括：
默认构造函数
参数为SharedPtr&的explicit单参数构造函数
拷贝构造函数
拷贝赋值函数
析构函数
lock()函数：取指向的SharePtr，如果未指向任何SharePtr，或者已被析构，返回指向nullptr的SharePtr
expired()函数：是否指向SharePtr，如果指向Share Ptr其是否已经析构
release()函数
*/
template<class T> class WeakPtr
{
public:
	friend class SharedPtr<T>;//方便weak_ptr与share_ptr设置引用计数和赋值。

	WeakPtr()
		: m_pResource(nullptr)
		, m_pCounter(nullptr)
	{
	}
	
	WeakPtr(SharedPtr<T>& other)
		: m_pResource(other.m_pResource)
		, m_pCounter(other.m_pCounter)
	{
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->weakCount);
		}
	}
	
	WeakPtr(WeakPtr<T>& other)
		: m_pResource(other.m_pResource)
		, m_pCounter(other.m_pCounter)
	{
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->weakCount);
		}
	}
	
	WeakPtr<T>& operator = (WeakPtr<T>& other)
	{
		if (this == &other) return *this;
	
		release();
	
		m_pCounter = other.m_pCounter;
		m_pResource = other.m_pResource;
	
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->weakCount);
		}
	
		return *this;
	}
	
	WeakPtr<T>& operator =(SharedPtr<T>& other)
	{
		release();
	
		m_pCounter = other.m_pCounter;
		m_pResource = other.m_pCounter;
	
		if (m_pCounter != nullptr)
		{
			++(m_pCounter->weakCount);
		}
	
		return *this;
	}
	
	~WeakPtr()
	{
		release();
	}
	
	SharedPtr<T> lock()
	{
		return SharedPtr<T>(*this);
	}
	
	bool expired()
	{
		if (m_pCounter != nullptr && m_pCounter->shareCount > 0)
			return false;
	
		return true;
	}

private:
	void release()
	{
		if (nullptr == m_PCounter) return;

		--m_pCounter->weakCount;
		if (0 == m_pCounter->weakCount && 0 == m_pCounter->shareCount) // 必须都为0才能删除
		{
			delete m_pCounter;
			m_pCounter = NULL;
		}
	}

private:
	T* m_pResource; // 可能会成为悬挂指针 此时m_pCounter->shareCount = 0
	Counter* m_pCounter;
};


#include <memory>

int main(void) {
	auto sp = std::make_shared<int>(42);
	std::weak_ptr<int> gw = sp;
	return 0;
}
```

## 7 虚函数的实现原理

虚函数表指针和虚函数表是C++实现多态的核心机制，理解vtbl和vptr的原理是理解C++对象模型的重要前提。
 class里面method分为两类：virtual 和non-virtual。非虚函数在编译器编译是静态绑定的，所谓静态绑定，就是编译器直接生成JMP汇编代码，对象在调用的时候直接跳转到JMP汇编代码执行，既然是汇编代码，那么就是不能在运行时更改的了；虚函数的实现是通过虚函数表，虚函数表是一块连续的内存，每个内存单元中记录一个JMP指令的地址，通过虚函数表在调用的时候才最终确定调用的是哪一个函数，这个就是动态绑定。

![img](https://i.loli.net/2021/06/26/KFfgQGMYCBOa63D.png)



![image-20210413225253856](https://i.loli.net/2021/06/26/Qwf6nFGXvyW2Ems.png)



 通过观察和测试，我们发现了以下几点问题：

1. 派生类对象d中也有一个虚表指针，d对象由两部分构成，一部分是父类继承下来的成员，虚表指针也就是存在部分的另一部分是自己的成员。
2. 基类b对象和派生类d对象虚表是不一样的，这里我们发现Func1完成了重写，所以d的虚表中存的是重写的Derive::Func1，所以**虚函数的重写也叫作覆盖，覆盖就是指虚表中虚函数的覆盖**。重写是语法的叫法，覆盖是原理层的叫法。
3. 另外Func2继承下来后是虚函数，所以放进了虚表，Func3也继承下来了，但是不是虚函数，所以不会放进虚表。
4. 虚函数表本质是一个存虚函数指针的指针数组，这个数组最后面放了一个nullptr。
5. 总结一下派生类的虚表生成： a.先将基类中的虚表内容拷贝一份到派生类虚表中 b.如果派生类重写了基类中某个虚函数，用派生类自己的虚函数覆盖虚表中基类的虚函数 c.派生类自己新增加的虚函数按其在派生类中的声明次序增加到派生类虚表的最后。

典型面试题

虚函数存在哪的？虚表存在哪的？ 答：**虚表存的是虚函数指针，不是虚函数，虚函数和普通函数一样的，都是存在代码段的，只是他的指针又存到了虚表中**。另外 **对象中存的不是虚表，存的是虚表指针**。



class的内部有一个virtual函数，其对象的首个地址就是vptr，指向虚函数表，虚函数表是连续的内存空间，也就是说，可以通过类似数组的计算，就可以取到多个虚函数的地址，还有一点，虚函数的顺序和其声明的顺序是一直的。



![img](https://i.loli.net/2021/07/25/U6BDyEsWg14XRqt.png)

> 怎么理解动态绑定和静态绑定，一般来说，对于类成员函数（不论是静态还是非静态的成员函数）都不需要创建一个在运行时的函数表来保存，他们直接被编译器编译成汇编代码，这就是所谓的静态绑定；所谓动态绑定就是对象在被创建的时候，在它运行的时候，其所携带的虚函数表，决定了需要调用的函数，也就是说，程序在编译完之后是不知道的，要在运行时才能决定到底是调用哪一个函数。这就是所谓的静态绑定和动态绑定。
> 参考: [C++this指针-百度百科](https://link.jianshu.com?t=http://baike.baidu.com/link?url=Yzd4GPwMhepMPfjjoAiQ6ZJgVNhLJ3QwjXoJzmcFMlh7JgI1nAt7iD7gyTqO-5IXHTNRPb1bs9njP_KdktnLvw2iXxTmOKJsZ9Sy3FifIoS_rCLVJIJtg2M9Oj8heK3m)

动态绑定需要三个条件同时成立：

> 1 指针调用
> 2 up-cast (有向上转型，父类指针指向子类对象)
> 3 调用的是虚函数

通过两张图看看汇编代码：



![img](https://i.loli.net/2021/07/25/LoxKkBwgjV1NyiA.png)

a.vfunc1()调用虚函数，那么a调用的是A的虚函数，还是B的虚函数？对象调用不会发生动态绑定，只有指针调用才会发生动态绑定。120行下面发生的call是汇编指令，call后面是一个地址，也就是函数编译完成之后的地址了。

再看第二张：

![img](https://i.loli.net/2021/04/13/9P2CpwAIElTNyRZ.png)

动态绑定

up-cast、指针调用、虚函数三个条件都满足动态调用，call指令后面不再是静态绑定简单的地址，翻译成C语言大概就是`(*(p->vptr)[n](p))`，通过虚函数表来调用函数。

参考链接：

https://cloud.tencent.com/developer/article/1688427



## 8 构造函数为什么不能声明为虚函数

虚函数需要依赖对象中指向类的虚函数表的指针，而这个指针是在构造函数中初始化的(这个工作是编译器做的，对程序员来说是透明的)，如果构造函数是虚函数的话，那么在调用构造函数的时候，而此时虚函数表指针并未初始化完成，这就会引起错误。



## 9 什么是字节对齐，为什么要采用这种机制？

一个存储区的地址必须是它自身大小的整数倍(double类型存储区的地址只需要是4的整数倍)，这个规则叫数据对齐

- 结构体成员合理安排位置，以节省空间
- 跨平台数据结构可考虑1字节对齐，节省空间但影响访问效率
- 跨平台数据结构人为进行字节填充，提高访问效率但不节省空间
- 本地数据采用默认对齐，以提高访问效率



无论数据是否对齐，大多数计算机还是能够正确工作，而且从前面可以看到，结构体test本来只需要11字节的空间，最后却占用了16字节，很明显**浪费了空间**，那么为什么还要进行字节对齐呢？最重要的考虑是**提高内存系统性能**
前面我们也说到，计算机每次读写一个字节块，例如，假设计算机总是从内存中取8个字节，如果一个double数据的地址对齐成8的倍数，那么一个内存操作就可以读或者写，但是如果这个double数据的地址没有对齐，数据就可能被放在两个8字节块中，那么我们可能需要执行两次内存访问，才能读写完成。显然在这样的情况下，是低效的。所以需要字节对齐来提高内存系统性能。
在有些处理器中，如果需要未对齐的数据，可能不能够正确工作甚至crash，这里我们不多讨论。



## 10 简述 STL 中的 map 的实现原理

stl的map是基于红黑树实现的

https://zhuanlan.zhihu.com/p/93917669



## 11 如果线上某台虚机CPU Load过高，该如何快速排查原因？只介绍思路和涉及的Linux命令即可 。

使用 top 命令 查找出cpu过高的进程



## 12 简述 C++ 中智能指针的特点，简述 new 与 malloc 的区别

智能指针它负责自动释放所指向的对象。

（1）new、delete 是操作符，可以重载，只能在 C++中使用。

（2）malloc、free 是函数，可以覆盖，C、C++中都可以使用。

（3）new 可以调用对象的构造函数，对应的 delete 调用相应的析构函数。

（4）malloc 仅仅分配内存，free 仅仅回收内存，并不执行构造和析构函数

（5）new、delete 返回的是某种数据类型指针，malloc、free 返回的是 void 指针。

注意：malloc 申请的内存空间要用 free 释放，而 new 申请的内存空间要用 delete 释放，不要混用。因为两者实现的机理不同。



## 13 简述 C++ 右值引用与转移语义

右值引用就是必须绑定到右值的引用，通过&&而不是&来获得右值引用。

右值引用有一个重要的性质，只能绑定到一个将要销毁的对象

算数表达式、后置递增递减运算符都生成右值，可以使用右值引用绑定到这类表达式上

右值引用是用来支持转移语义的。**转移语义可以将资源 ( 堆，系统对象等 ) 从一个对象转移到另一个对象**，这样能够减少不必要的临时对象的创建、拷贝以及销毁，能够大幅度提高 C++ 应用程序的性能。临时对象的维护 ( 创建和销毁 ) 对性能有严重影响



## 14 简述 vector 的实现原理

vector底层与Array一样都是连续的内存空间，区别在于vector的空间是动态的，随着更多元素的加入可以自动实现空间扩展，并且vector针对这种扩展做了优化，并不是one by one的扩展，那样实在是低效，而是按照某种倍率来扩展，这样就有效了减少因为扩容带来的复制效率降低问题。

简单来说就是当需要放置1个元素时，vector空间已满，此时vector并不会只向系统申请1个元素的空间，而是按照目前已占用的空间的倍率来申请。

假如原来占用A字节，那么再次申请时可能是2A字节，由于此时向尾部地址扩展不一定有连续未分配的内存，大多时候还是会涉及`开辟新的更大空间、将旧空间元素复制到新空间、释放旧空间`三个大步骤。

所以和数组相比底层的操作都是一样的，不要把Vector神话，都是普通的结构只不过被封装了一层而已。

从本质上看，vector就是在普通Array和使用者中间加了一层，从而把使用者从对数组的直接管理权接手过来，让使用者有个管家一样，在毫无影响使用的前提下更加省心和高效。



## 15 C++ 中智能指针和指针的区别是什么？

智能指针的行为类似常规指针，重要的区别是它负责自动释放所指向的对象。

标准库中新加shared_ptr,unique_ptr智能指针

shared_ptr允许多个指针指向同一个对象，unique_ptr则独占所指向的对象

```
auto p = make_shared<vector< string>>();
```

每个shared_ptr都有一个关联的计数器，通过称为引用计数，当我们拷贝一个shared_ptr，计数器都会递增，

当局部的shared_ptr离开其作用域，计数器会递减

一旦一个shared_ptr的计数器变为0，它就会自动释放自己所管理的对象



## 16 C++ 中多态是怎么实现的

什么是多态呢？就是程序运行时，父类指针可以根据具体指向的子类对象，来执行不同的函数，表现为多态。

- 存在虚函数时，编译器会为对象自动生成一个指向虚函数表的指针（通常称之为 vptr 指针）
- 调用此类的构造函数时，在类的构造函数中，编译器会隐含执行 vptr 与 vtable 的关联代码，将 vptr 指向对应的 vtable，将类与此类的 vtable 联系了起来
- 指向基类的指针此时已经变成指向子类的 this 指针，这样依靠此 this 指针即可得到正确的 vtable，。如此才能真正与函数体进行连接，实现多态的基本原理。



## 17 const常量与 #define定义的常量有什么区别？

1)   类型：const 常量有类型，#define定义的常量没有类型

2)   作用不同：const定义一个不可更改的量，#define给常量定义一个宏名

##   18 extern “C”有什么作用?

屏蔽c++名称修饰机制，可以兼容c程序



## 19 C++ 11 有什么新特性

1 类型推导auto

2 decltype，decltype操作符的值是一个类型，可用于其他对象的声明

3 using取类型别名

4 连续出现的右尖括号不会再被误以为是右移运算符,可以使用小括号

5 初始列表，数据类型 变量 { 初始化列表 }

6 变长初始化表initializer_list，initializer_list作为一个轻量级的链表容器，不但可以用在构造函数中，也可以作为普通函数的参数，传递不同数量的实参；

轻量级容器内部存放初始化列表元素的引用而非其拷贝，重量级容器内部存放初始化列表元素的拷贝而非其引用

7 基于范围的for循环

8  函数绑定bind 

​		A a;
​		auto f1 = bind (&A::foo, &a);

9 lambda表达式

[ 捕获表 ]（参数表） 选项 ->  返回类型 { 函数体 }

10 右值引用

11 泛型元组tuple

12 Variadic Templates，可变参数模板，完成递归函数，递归继承和STL的一些实现



## 20 C++ 中智能指针和指针的区别是什么？

智能智能是模板类，离开作用域可以自动释放所指向的资源，普通指针需要手动释放指针所指向的资源



## 22 实现一个线程安全的队列

```c++
/*
实现一个线程安全的队列（多生产者多消费者）
*/

template <class T>
class SafeQueue
{
public:
	SafeQueue(void):q(),m(),c()
	{}
	~SafeQueue(void)
	{}
	// Add an element to the queue.
	void enqueue(T t)
	{
		std::lock_guard<std::mutex> lock(m);
		q.push(t);
		c.notify_one();
	}

	// Get the "front"-element.

 // If the queue is empty, wait till a element is avaiable.
	T dequeue(void)
	{
		std::unique_lock<std::mutex> lock(m);
		
        //wait,只有q队列非空时（有任务可以执行），继续往下执行
		//c.wait(lock);直接写成这样，有可能导致虚假唤醒发生，尽量避免
		//可以写成这种代替下面循环 c.wait(lock, [=] {return  !q.empty(); });   
        while (q.empty())
		{
			// release lock as long as the wait and reaquire it afterwards.
			c.wait(lock);
		}
		T val = q.front();
		q.pop();
		return val;
	}

private:
	std::queue<T> q;
	mutable std::mutex m;
	std::condition_variable c;
};
```



## 23 重新实现一个更“优”的 string

要求： 

(0) 对外的行为表现与std::string 完成一致

(1) 优化点是：复制的时候仅复制引用，只有在修改内容时，才复制内容

线程安全

str = 'abcde'

char = str.read(i, 'c')

str.write(j, 'd')

strb = stra

stra[1] = 'a'

strb[2] = 'b'



```c++
/*
优化string
复制的时候 仅复制引用，只有在修改内容时，才复制内容
即实现写时拷贝
*/

class COWMyString
{
public:
	//默认参数
	COWMyString(const char *str = "") :m_str(strcpy(new char[strlen(str) + 1], str))
	{
		if (m_str) {
			m_count = new size_t(1);
		}
		else {
			m_count = new size_t(0);
		}
	}

	~COWMyString(void)
	{
	
		(*this->m_count)--;
		if (*this->m_count == 0)
		{
			delete []this->m_str;
			delete this->m_count;
		}
	}
	
	//深拷贝构造
	COWMyString(const COWMyString&that)
	{
		if (this != &that) {
			this->m_str = that.m_str;
			//指针指向的实际上是同一个计数器，同一块内存
			this->m_count = that.m_count;
			(*this->m_count)++;
		}
	}
	
	//深拷贝赋值
	COWMyString&operator=(const COWMyString&that)
	{
		//指针指向的实际上是同一个计数器，同一块内存
		if (this->m_str == that.m_str)
		{
			return *this;
		}
	
		if (this->m_str)
		{
			//将原来的引用计数减一
			(*this->m_count)--;
			if (this->m_count == 0) {
				delete []this->m_str;
				delete this->m_count;
			}
	
			this->m_str = that.m_str;
			//指针指向的实际上是同一个计数器，同一块内存
			this->m_count = that.m_count;
			(*this->m_count)++;
		}
	
		return *this;
	}
	
	//copy on write
	char &operator[](size_t index)
	{
		if (index>strlen(m_str))
		{
			static char nullchar = 0;
			return nullchar;
		}
		(*this->m_count)--;
		m_str = strcpy(new char[strlen(m_str)+1], m_str);
		m_count = new size_t(1);
	
		return *(m_str + index);
	
	}
	
	const char *c_str(void)const
	{
		return m_str;
	}

private:
	//指针
	char* m_str;
	//计数器
	size_t* m_count;
};
```



## 24 表达式计算

"ADD(1,1)" => 2

"SUB(1110, 0)" => 1110

"ADD(SUB(20,1),ADD(1,1))" => 21

"ADD(SUB(10,ADD(1,1)), 10)" => 18









## 25 C++实现string的简单功能

```c++
/*
实现一个string满足基本用法
*/
//class定义的类默认的访问控制属性为private，而struct定义的类默认访问属性是public
class MyString
{
public:
	//默认参数
    /*初始化表 ：（1）如果有类 类型的成员变量，而该类又没有无参构造函数，则必须通过初始化表来初始化该变量
    		   （2）如果类中包含“const"或"引用&"成员变量，必须在初始表中进行初始化。
   	对象的创建
  --》为对象分配内存
  --》调用成员子对象的构造函数(声明顺序),在初始化表中初始化按照声明顺序
  --》执行构造函数的代码
  注意：先执行初始化表，后执行构造函数
   	*/
	MyString(const char *str=""):m_str(strcpy(new char[strlen(str)+1], str))
	{

	}
	
    /*
    如果一个类没有显示定义析构函数，那么编译器会为该类提供一个缺省的析构函数；
		1）对基本类型的成员变量，什么也不做
		2）对类类型的成员变量，调用相应类型的析构函数
	对象的销毁
  --》执行析构函数的代码
  --》调用成员子对象的析构函数(声明逆序) 
  --》释放对象的内存空间
    */
	~MyString(void)
	{
		if (m_str)
		{//数组形式的删除内存
			delete []m_str;  
			m_str = nullptr;
		}
	}
	
	//深拷贝构造
    /*
     如果一个类包含指针形式的成员变量，缺省拷贝构造函数只是复制了指针成员变量本身，而没有复制指针所指向数据，这种拷贝称为浅拷贝。
     浅拷贝将导致不同对象间的数据共享，如果数据在堆区，会在析构时引发“double free”异常.
     为此必须自己定义一个支持复制指针所指向内容的拷贝构造函数，即深拷贝。
    */
	MyString(const MyString&that):m_str(strcpy(new char[strlen(that.m_str) + 1], that.m_str))
	{
	
	}
	
	//深拷贝赋值
    //深拷贝赋值返回的都是MyString&
	MyString&operator=(const MyString&that)
	{
		//防止紫赋值
		if (&that!=this)
		{
			MyString temp(that);//深拷贝构造,temp是i2的临时对象
			swap(m_str, temp.m_str); 
		}
	
		return  *this;
	}
    
    //常函数
    /*
    在一个成员函数的参数后面加上const，这个成员函数就称为常函数。
    常函数中的this指针是一个常指针，不能在常函数中修改成员变量的值。
    被mutable关键字修饰的成员变量可以在常函数中被修改
    非常对象既可以调用非常函数，也可以调用常函数，但是常对象只能调用常函数，不能调用非常函数。
    函数名和形参表相同的成员函数，其常版本和非常版本，可以构成重载关系，常对象调用常版本，非常调用调用非常版本。
    */
	const char *c_str(void)const
	{
		return m_str;
	}

private:
	char* m_str;
};
```



## 26 实现线程安全的单例模式

```c++
/*
实现一个线程安全的单例模式
*/

class A
{
public:
	//静态函数，返回引用
	static A &GetInstance()
	{//静态局部变量
		static A s_instance;
		return s_instance;
	}
private:
	//默认构造函数
    //缺省构造函数，如果一个类没有定义构造函数，编译器会为其提供一个缺省的构造函数(无参构造函数)。
    //对于基本类型的成员变量不做初始化
    //对类类型的成员变量，用相应类型的无参构造函数初始化
	A() = default;
	/*
	拷贝构造函数
		用一个已存在对象构造同类型的副本对象时，会调用拷贝构造函数。
		class 类名{
		public:
			类名(const 类名& that){...}
		};
	*/
	A(const A &that) = delete;				//禁止使用拷贝构造函数
	A& operator=(const A&that) = delete;	//禁止使用拷贝赋值用算符
};
```



## 27 纯虚函数 VS 虚函数

```c++
//纯虚函数
virtual void funtion1()=0
```



1. 纯虚函数没有定义，普通虚函数必须有定义
2. 包含纯虚函数的类为抽象类，不能生成对象；包含虚函数的类可以生成对象
3. 如果基类中包含纯虚函数，则必须在子类中重新实现函数该函数（不能写成=0的形式，否则子类也是一个抽象类，不能生成对象）
4. 定义纯虚函数的主要目的是为了实现一个接口，继承这个类的程序员必须实现这个函数
5. 抽象类不能定义实例，但可以声明指向实现该抽象类的具体类的指针或引用
6. 当基类中的某个成员函数必须由子类提供个性化实现的时候，应该设计为纯虚函数；当基类中某个成员函数大多数情况下都应该由子类提供个性化实现，但基类也可以提供缺省实现的时候，应该设计为普通虚函数



## lamda表达式

[ 捕获表 ]（参数表） 选项 -> 返回类型 { 函数体 }

https://www.cnblogs.com/LuckCoder/p/8668125.html

## std:::thread 基础用法

https://www.cnblogs.com/LuckCoder/p/11436100.html

thread th1(ThreadFunc1);

使用全局函数作为线程函数 

==================================

  Foo foo;
  thread th1(&Foo::ThreadFunc1,foo);

使用类成员函数作为线程函数

=======================================

 thread th([] {while (1)
    {
        std::cout << "lambda" << endl;
    }
    });

使用lambda表达式作为线程函数

## std::future 用法

使用std::async创建简单异步任务

https://baptiste-wicht.com/posts/2017/09/cpp11-concurrency-tutorial-futures.html

## C++类型转换

### static_cast

任何具有明确定义的类型转换，只要不包含底层const,都可以使用static_cast

example:

```c++
int j=10;
double value=static_cast<double>(j)/2; 
```

```c++
void *p =&d;
//必须确保d是double类型，否则类型不符合，会产生未定义的结果
double *dp=static_cast<double *>(p);
```

### const_cast

const_cast只能改变运算对象的底层const，即将常量对象转换成非常量对象。去掉const性质

const_cast常常用于函数重载种，如一个函数是常函数，一个是普通函数，则可以直接复用一份代码

注：如果对象是常量，再使用const_cast执行操作就会产生未定义的行为

```c++
const volatile int ci = 100;
int* pci = const_cast<int*>(&ci);
```

### reinterpret_cast

reinterpret_cast通常为运算对象的位模式提供较低层次上的重新解释

```c++
int *ip;
char *pc = reinterpret_cast<char *>(ip); //编译器不会报错
string str(pc);//可能在运行时发生错误，pc所值的真实对象是一个int而非字符
```

使用reinterpret_cast非常危险，一般不建议使用

### dynamic_cast

dynamic_cast用于将基类的指针或引用安全的转换成派生类的指针或引用

使用场景：

想使用基类的指针或者引用执行子类的某个函数并且该函数不是虚函数（一般情况，我们应该尽量使用虚函数）

当基类指针指向子类对象时，dynamic_cast是安全的并且可以执行转换，否则会执行失败或则抛出异常

```c++
//A是基类，B是子类
B b;
A* pa = &b;
B* pb = dynamic_cast<B*>(pa);
```





## 静态编译VS 动态编译

### 静态编译与动态编译的区别

静态编译就是在编译时，把所有模块都编译进可执行文件里，当启动这个可执行文件时，所有模块都被加载进来；

动态编译是将应用程序需要的模块都编译成动态链接库，启动程序（初始化）时，这些模块不会被加载，运行时用到哪个模块就调用哪个

### 静态库

优点：
 代码的装载速度快，执行速度也比较快
 缺点：

1. 程序体积会相对大一些
2. 如果静态库需要更新，程序需要重新编译
3. 如果多个应用程序使用的话，会被装载多次，浪费内存。

### 动态库

动态链接库：在应用程序启动的时候才会链接，所以，当用户的系统上没有该动态库时，应用程序就会运行失败
 优点：
 多个应用程序可以使用同一个动态库，而不需要在磁盘上存储多个拷贝

缺点：
 由于是运行时加载，可能会影响程序的前期执行性能

>>>>>>> b20f09a58c7d19d41deecee8fbc5198639cf8e88



## 1 为什么要有初始化表？

1 如果没有在构造函数的初始值列表中显示地初始化成员，则该成员在构造函数体之前执行默认初始化（类类型的参数执行默认构造函数，普通类型的参数值为随机数），即先定义然后在构造函数体内赋值。

2 如果成员时const或者引用的话，必须将其初始化，因为引用和const变量必须初始化，不能先定义后赋值

3 当成员属于某种类类型且该类没有定义默认构造函数时（提供的构造函数均为有参构造函数），必须将这个成员初始化，因为无法对该成员的构造函数进行调用

注：如果一个构造函数为所有参数提供了默认实参，则它实际上也定义了默认构造函数

4 成员变量的初始化顺序由声明顺序决定，而与初始化表的顺序无关，最好令构造函数初始值的顺序与成员声明的顺序一致，最好令构造函数的参数作为成员的初始值，尽量避免使用其他成员初始化其他成员。

5 对象的创建过程：

（1）为对象分配内存

（2）调用成员子对象的构造函数(声明顺序)

（3）执行构造函数的代码



## 2 基类的析构函数为什么要设计为虚函数？

基类中的析构函数如果是虚函数，那么派生类的析构函数就重写了基类的析构函数。这里他们的函数名不相同，看起来违背了重写的规则，但实际上编译器对析构函数的名称做了特殊处理，编译后析构函数的名称统一处理成destructor。

子类的析构函数，会自动调用基类的析构函数，析构基类子对象。

注：析构函数的运作方式是最深层派生的那个子类其析构函数最先被调用，然后是其每一个基类的析构函数被调用，编译器会在子类的析构函数中创建一个对基类的析构函数的调用动作，所以基类的析构函数必须有定义。

基类的析构函数不会调用子类的析构函数，对一个指向子类对象的基类指针使用delete运算符，实际被执行的只有基类的析构函数，所释放的仅仅是基类子对象的动态资源，如果子类中有其它的动态资源分配将会形成内存泄露。

将基类的析构函数设计为虚函数，可利用多态调用子类的析构函数，进而调用基类的析构函数，释放所有资源。



## 3  智能指针有几种（shared_ptr,unique_ptr,weakPtr,autoptr）？每种智能指针的实现原理？每种智能指针的适用场景？为什么要有weakPtr？

智能指针是一个模板类，当离开作用域时，可以调用其析构函数自动释放内存资源。

智能指针有三种，shared_ptr，unique_ptr,weak_ptr。

shared_ptr允许多个指针指向同一个对象，unique_ptr则独占所指向的对象，weak_ptr是一种弱引用，指向shared_ptr所管理的对象。

shared_ptr的实现原理是在其内部维护了一个计数器（也是动态分配的内存），当我们拷贝一个shared_ptr时，计数器会递增，当我们给shared_ptr赋予一个新值或是shared_ptr被销毁时，计数器会递减。

unique_ptr的实现原理是拷贝构造和赋值运算符都被删除（或者定义为私有），保留了移动构造和移动赋值函数。

weak_ptr的实现原理是在其内部维护了一个weak_count计数器。

autoptr是C++11之前实现的智能指针，现在已经基本弃用，有两个明显缺陷：第一个是使用delete操作符释放资源，如果是动态分配的数组对象，无法进行资源释放，第二个是两个指针不能指向相同的对象，当进行赋值操作时，会将另一个指针置为空值。

shared_ptr和weak_ptr拥有相同的基类，在这个基类中维护了引用计数基类_Ref_count_base，这个基类中有两个数据成员，_Uses和_Weaks，

```c++
_Atomic_counter_t _Uses;
_Atomic_counter_t _Weaks;
```

_Uses则是我们理解的引用计数，每有一个shared_ptr共享资源，_Uses就会加一，反之每一个shared_ptr析构，_Uses就会减一，当_Uses变为零时，意味着没有shared_ptr再占有资源，这个时候占有的资源会调用释放操作。但是并不能直接销毁引用计数对象，因为可能有弱引用还绑定到引用计数对象上。

而_Weaks就是weak_ptr追踪资源的计数器，每有一个weak_ptr追踪资源，_Weaks就会加一，反之每一个weak_ptr析构时，_Weaks就会减一，当_Weaks变为零时，意味着没有weak_ptr再追踪资源，这时会销毁引用计数对象。（可使用VS21017查看智能指针的实现源码）

三种智能指针使用C++的简单实现代码参考如下：

https://github.com/Mr-jiayunfei/code_learn/blob/main/STL/implement_SmartPtr/main.cpp



weak_ptr的使用场景

weak_ptr只能从shared_ptr对象构建。

weak_ptr并不影响动态对象的生命周期，即其存在与否并不影响对象的引用计数器。当weak_ptr所指向的对象因为shared_ptr计数器为0而被释放后，那么weak_ptr的lock方法将返回空。

weak_ptr并没有重载operator->和operator *操作符，因此不可直接通过weak_ptr使用对象。

提供了expired()与lock()成员函数，前者用于判断weak_ptr指向的对象是否已被销毁，后者返回其所指对象的shared_ptr智能指针(对象销毁时返回”空“shared_ptr)，如果返回shared_ptr，那么计数器会加1.

`std::weak_ptr`是解决悬空指针问题的一种很好的方法。仅通过使用原始指针，不知道所引用的数据是否已被释放。

相反，通过`std::shared_ptr`管理数据并提供`std::weak_ptr`给用户，用户可以通过调用`expired()`或来检查数据的有效性`lock()`。

weak_ptr可以解决shared_ptr的循环引用问题



## 4 虚函数的实现原理

虚函数表指针和虚函数表是C++实现多态的核心机制，理解vtbl和vptr的原理是理解C++对象模型的重要前提。
 class里面method分为两类：virtual 和non-virtual。非虚函数在编译器编译是静态绑定的，所谓静态绑定，就是编译器直接生成JMP汇编代码，对象在调用的时候直接跳转到JMP汇编代码执行，既然是汇编代码，那么就是不能在运行时更改的了；虚函数的实现是通过虚函数表，虚函数表是一块连续的内存，每个内存单元中记录一个JMP指令的地址，通过虚函数表在调用的时候才最终确定调用的是哪一个函数，这个就是动态绑定。

![img](https://i.loli.net/2021/04/13/zYtX8ZRkHnES31l.png)



![image-20210413225253856](https://i.loli.net/2021/04/13/DX5PJxjtwqRgbsG.png)



 通过观察和测试，我们发现了以下几点问题：

1. 派生类对象d中也有一个虚表指针，d对象由两部分构成，一部分是父类继承下来的成员，虚表指针也就是存在部分的另一部分是自己的成员。
2. 基类b对象和派生类d对象虚表是不一样的，这里我们发现Func1完成了重写，所以d的虚表中存的是重写的Derive::Func1，所以**虚函数的重写也叫作覆盖，覆盖就是指虚表中虚函数的覆盖**。重写是语法的叫法，覆盖是原理层的叫法。
3. 另外Func2继承下来后是虚函数，所以放进了虚表，Func3也继承下来了，但是不是虚函数，所以不会放进虚表。
4. 虚函数表本质是一个存虚函数指针的指针数组，这个数组最后面放了一个nullptr。
5. 总结一下派生类的虚表生成： a.先将基类中的虚表内容拷贝一份到派生类虚表中 b.如果派生类重写了基类中某个虚函数，用派生类自己的虚函数覆盖虚表中基类的虚函数 c.派生类自己新增加的虚函数按其在派生类中的声明次序增加到派生类虚表的最后。

典型面试题

虚函数存在哪的？虚表存在哪的？ 答：**虚表存的是虚函数指针，不是虚函数，虚函数和普通函数一样的，都是存在代码段的，只是他的指针又存到了虚表中**。另外 **对象中存的不是虚表，存的是虚表指针**。



class的内部有一个virtual函数，其对象的首个地址就是vptr，指向虚函数表，虚函数表是连续的内存空间，也就是说，可以通过类似数组的计算，就可以取到多个虚函数的地址，还有一点，虚函数的顺序和其声明的顺序是一直的。



![img](https://i.loli.net/2021/07/25/U6BDyEsWg14XRqt.png)

> 怎么理解动态绑定和静态绑定，一般来说，对于类成员函数（不论是静态还是非静态的成员函数）都不需要创建一个在运行时的函数表来保存，他们直接被编译器编译成汇编代码，这就是所谓的静态绑定；所谓动态绑定就是对象在被创建的时候，在它运行的时候，其所携带的虚函数表，决定了需要调用的函数，也就是说，程序在编译完之后是不知道的，要在运行时才能决定到底是调用哪一个函数。这就是所谓的静态绑定和动态绑定。
> 参考: [C++this指针-百度百科](https://link.jianshu.com?t=http://baike.baidu.com/link?url=Yzd4GPwMhepMPfjjoAiQ6ZJgVNhLJ3QwjXoJzmcFMlh7JgI1nAt7iD7gyTqO-5IXHTNRPb1bs9njP_KdktnLvw2iXxTmOKJsZ9Sy3FifIoS_rCLVJIJtg2M9Oj8heK3m)

动态绑定需要三个条件同时成立：

> 1 指针调用
> 2 up-cast (有向上转型，父类指针指向子类对象)
> 3 调用的是虚函数

通过两张图看看汇编代码：



![img](https://i.loli.net/2021/07/25/LoxKkBwgjV1NyiA.png)

a.vfunc1()调用虚函数，那么a调用的是A的虚函数，还是B的虚函数？对象调用不会发生动态绑定，只有指针调用才会发生动态绑定。120行下面发生的call是汇编指令，call后面是一个地址，也就是函数编译完成之后的地址了。

再看第二张：

![img](https://i.loli.net/2021/04/13/9P2CpwAIElTNyRZ.png)

动态绑定

up-cast、指针调用、虚函数三个条件都满足动态调用，call指令后面不再是静态绑定简单的地址，翻译成C语言大概就是`(*(p->vptr)[n](p))`，通过虚函数表来调用函数。

参考链接：

https://cloud.tencent.com/developer/article/1688427

## 5 构造函数为什么不能声明为虚函数

虚函数需要依赖对象中指向类的虚函数表的指针，而这个指针是在构造函数中初始化的(这个工作是编译器做的，对程序员来说是透明的)，如果构造函数是虚函数的话，那么在调用构造函数的时候，而此时虚函数表指针并未初始化完成，这就会引起错误。





# linux下使用INT_MAX宏需要包含 limits.h头文件

```
#include <limits.h>
```

# C++判读一个数是否是平方数

```c++
#include <math.h>

int main(){

   double root;

   root = sqrt(200);

​  printf("answer is %f\n", root);

}
```



# C++的取整 向下取整，向上取整，四舍五入取整，直接去小数点取整

作用							函数名称							函数说明									2.1	2.9	-2.1	-2.9
向下取整					floor()								不大于自变量的最大整数			2	2	-3	-3
向上取整					ceil()									不小于自变量的最大整数			3	3	-2	-2
四舍五入取整			自定义round()					四舍五入到最邻近的整数			2	3	-2	-3
直接去小数点取整	int(double a)					直接从小数转整数，去小数点	2	2	-2	-2


其中四舍五入取整一般需要自己处理，也就是看小数点第一位，所以对于正数而言，加上0.5后，向下取整；对于负数而言，减去0.5，向下取整，具体代码如下：

```c++
int round(double r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}
```


