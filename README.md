# Data Structures & Algorithms in Swift
## Your Complete Guide to Mastering DSA

---

## Table of Contents

### Part 1: Foundations
1. [Introduction & Complexity Analysis](#chapter-1)
2. [Swift Fundamentals for DSA](#chapter-2)
3. [Problem-Solving Patterns](#chapter-3)

### Part 2: Arrays & Strings
4. [Arrays Deep Dive](#chapter-4)
5. [String Manipulation](#chapter-5)
6. [Two Pointers Technique](#chapter-6)
7. [Sliding Window Pattern](#chapter-7)

### Part 3: Linked Lists
8. [Singly Linked Lists](#chapter-8)
9. [Doubly Linked Lists](#chapter-9)
10. [Advanced Linked List Problems](#chapter-10)

### Part 4: Stacks & Queues
11. [Stack Implementation & Problems](#chapter-11)
12. [Queue Implementation & Problems](#chapter-12)
13. [Monotonic Stack/Queue](#chapter-13)

### Part 5: Hash Tables & Sets
14. [Hash Tables Deep Dive](#chapter-14)
15. [Set Operations](#chapter-15)

### Part 6: Trees
16. [Binary Trees](#chapter-16)
17. [Binary Search Trees](#chapter-17)
18. [Tree Traversals](#chapter-18)
19. [Advanced Tree Problems](#chapter-19)

### Part 7: Heaps & Priority Queues
20. [Heap Implementation](#chapter-20)
21. [Priority Queue Problems](#chapter-21)

### Part 8: Graphs
22. [Graph Representations](#chapter-22)
23. [BFS & DFS](#chapter-23)
24. [Advanced Graph Algorithms](#chapter-24)

### Part 9: Sorting & Searching
25. [Sorting Algorithms](#chapter-25)
26. [Binary Search Mastery](#chapter-26)

### Part 10: Advanced Topics
27. [Dynamic Programming](#chapter-27)
28. [Backtracking](#chapter-28)
29. [Greedy Algorithms](#chapter-29)
30. [Bit Manipulation](#chapter-30)

---

<a name="chapter-1"></a>
## Chapter 1: Introduction & Complexity Analysis

### Why Data Structures & Algorithms?

Data structures and algorithms are the foundation of software engineering. They help you:
- Write efficient, scalable code
- Solve complex problems systematically
- Ace technical interviews
- Build better products

### Big O Notation

Big O describes how the runtime or space requirements grow as input size increases.

#### Common Time Complexities (Best to Worst)

| Notation | Name | Example |
|----------|------|---------|
| O(1) | Constant | Array access by index |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Array traversal |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Nested loops |
| O(2ⁿ) | Exponential | Recursive fibonacci |
| O(n!) | Factorial | Permutations |

#### Visual Representation

```
Time
 ^
 |                                            O(n!)
 |                                        O(2ⁿ)
 |                                   O(n²)
 |                          O(n log n)
 |                    O(n)
 |              O(log n)
 |        O(1)
 |________________________> Input Size (n)
```

### Analyzing Time Complexity

#### Rule 1: Drop Constants
```swift
// Both are O(n), not O(2n) or O(3n)
func example1(_ arr: [Int]) {
    for num in arr { print(num) }      // O(n)
    for num in arr { print(num * 2) }  // O(n)
}
// Total: O(n) + O(n) = O(2n) → O(n)
```

#### Rule 2: Drop Non-Dominant Terms
```swift
// O(n² + n) → O(n²)
func example2(_ arr: [Int]) {
    for i in arr {              // O(n)
        print(i)
    }
    for i in arr {              // O(n)
        for j in arr {          // O(n)
            print(i, j)
        }
    }
}
// Total: O(n + n²) → O(n²)
```

#### Rule 3: Different Inputs Use Different Variables
```swift
// O(a + b), NOT O(n)
func example3(_ arr1: [Int], _ arr2: [Int]) {
    for num in arr1 { print(num) }  // O(a)
    for num in arr2 { print(num) }  // O(b)
}

// O(a * b), NOT O(n²)
func example4(_ arr1: [Int], _ arr2: [Int]) {
    for num1 in arr1 {              // O(a)
        for num2 in arr2 {          // O(b)
            print(num1, num2)
        }
    }
}
```

### Space Complexity

Space complexity measures memory usage as input grows.

```swift
// O(1) space - constant extra memory
func sum(_ arr: [Int]) -> Int {
    var total = 0  // Only one variable
    for num in arr {
        total += num
    }
    return total
}

// O(n) space - memory grows with input
func double(_ arr: [Int]) -> [Int] {
    var result = [Int]()  // New array of size n
    for num in arr {
        result.append(num * 2)
    }
    return result
}

// O(n) space - recursive call stack
func factorial(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)  // n calls on stack
}
```

### Problem-Solving Framework

Follow this approach for every problem:

1. **Understand**: Clarify inputs, outputs, and edge cases
2. **Plan**: Choose data structures and algorithms
3. **Implement**: Write clean, readable code
4. **Test**: Verify with examples and edge cases
5. **Optimize**: Analyze complexity and improve

### Practice Problem: Two Sum

**Problem**: Find two numbers in array that add up to target.

```swift
// ❌ Brute Force - O(n²) time, O(1) space
func twoSumBrute(_ nums: [Int], _ target: Int) -> [Int]? {
    for i in 0..<nums.count {
        for j in (i+1)..<nums.count {
            if nums[i] + nums[j] == target {
                return [i, j]
            }
        }
    }
    return nil
}

// ✅ Optimal - O(n) time, O(n) space
func twoSum(_ nums: [Int], _ target: Int) -> [Int]? {
    var seen = [Int: Int]()  // value: index
    
    for (i, num) in nums.enumerated() {
        let complement = target - num
        if let j = seen[complement] {
            return [j, i]
        }
        seen[num] = i
    }
    return nil
}

// Test
print(twoSum([2, 7, 11, 15], 9) ?? [])  // [0, 1]
print(twoSum([3, 2, 4], 6) ?? [])       // [1, 2]
```

**Key Insight**: Trade space for time using a hash table!

<a name="chapter-2"></a>
## Chapter 2: Swift Fundamentals for DSA

### Essential Swift Features

#### Value Types vs Reference Types

```swift
// Value Types (struct, enum) - Copied
struct Point {
    var x: Int
    var y: Int
}

var p1 = Point(x: 1, y: 2)
var p2 = p1
p2.x = 10
print(p1.x)  // 1 (unchanged)
print(p2.x)  // 10

// Reference Types (class) - Shared
class Node {
    var value: Int
    var next: Node?
    init(_ value: Int) { self.value = value }
}

let n1 = Node(1)
let n2 = n1
n2.value = 10
print(n1.value)  // 10 (changed!)
print(n2.value)  // 10
```

**DSA Tip**: Use `class` for linked structures (trees, graphs). Use `struct` for simple data containers.

#### Optionals & Safety

```swift
// Unwrapping optionals safely
var arr = [1, 2, 3]

// Optional binding
if let first = arr.first {
    print(first)
}

// Nil coalescing
let value = arr.first ?? 0

// Guard statements
func process(_ nums: [Int]?) {
    guard let nums = nums, !nums.isEmpty else {
        return
    }
    // Work with nums safely
}
```

#### Inout Parameters (Pass by Reference)

```swift
// Useful for in-place modifications
func swap(_ a: inout Int, _ b: inout Int) {
    let temp = a
    a = b
    b = temp
}

var x = 5, y = 10
swap(&x, &y)
print(x, y)  // 10 5
```

#### Closures

```swift
// Sorting with custom comparator
let nums = [3, 1, 4, 1, 5, 9, 2, 6]

// Ascending
let sorted1 = nums.sorted { $0 < $1 }

// Descending
let sorted2 = nums.sorted { $0 > $1 }

// Custom comparison
struct Person {
    let name: String
    let age: Int
}

let people = [
    Person(name: "Alice", age: 30),
    Person(name: "Bob", age: 25)
]

let byAge = people.sorted { $0.age < $1.age }
```

### Useful Swift Collections

#### Arrays
```swift
var arr = [1, 2, 3]

// Adding elements
arr.append(4)           // [1, 2, 3, 4]
arr.insert(0, at: 0)    // [0, 1, 2, 3, 4]
arr += [5, 6]           // [0, 1, 2, 3, 4, 5, 6]

// Removing elements
arr.removeLast()        // [0, 1, 2, 3, 4, 5]
arr.remove(at: 0)       // [1, 2, 3, 4, 5]
arr.removeAll()         // []

// Access
arr = [1, 2, 3, 4, 5]
let first = arr.first   // Optional(1)
let last = arr.last     // Optional(5)
let slice = arr[1...3]  // [2, 3, 4]
```

#### Dictionaries
```swift
var dict = [String: Int]()

// Adding/updating
dict["apple"] = 5
dict["banana"] = 3

// Safe access
if let count = dict["apple"] {
    print(count)
}

// Default value
let oranges = dict["orange", default: 0]

// Iteration
for (key, value) in dict {
    print("\(key): \(value)")
}
```

#### Sets
```swift
var set: Set<Int> = [1, 2, 3, 3]  // {1, 2, 3}

// Operations
set.insert(4)           // {1, 2, 3, 4}
set.remove(2)           // {1, 3, 4}
set.contains(3)         // true

// Set operations
let set1: Set = [1, 2, 3]
let set2: Set = [2, 3, 4]

set1.union(set2)        // {1, 2, 3, 4}
set1.intersection(set2) // {2, 3}
set1.subtracting(set2)  // {1}
```

### Common Swift Patterns for DSA

#### Range Operations
```swift
// Creating ranges
for i in 0..<5 { }      // 0, 1, 2, 3, 4
for i in 0...5 { }      // 0, 1, 2, 3, 4, 5
for i in stride(from: 0, to: 10, by: 2) { }  // 0, 2, 4, 6, 8

// Reversed
for i in (0..<5).reversed() { }  // 4, 3, 2, 1, 0
```

#### Enumerated Loops
```swift
let arr = ["a", "b", "c"]

for (index, value) in arr.enumerated() {
    print("\(index): \(value)")
}
// 0: a
// 1: b
// 2: c
```

#### Functional Methods
```swift
let nums = [1, 2, 3, 4, 5]

// map - transform elements
let doubled = nums.map { $0 * 2 }  // [2, 4, 6, 8, 10]

// filter - keep matching elements
let evens = nums.filter { $0 % 2 == 0 }  // [2, 4]

// reduce - combine to single value
let sum = nums.reduce(0, +)  // 15

// compactMap - map + remove nils
let strings = ["1", "2", "a", "3"]
let numbers = strings.compactMap { Int($0) }  // [1, 2, 3]
```

<a name="chapter-3"></a>
## Chapter 3: Problem-Solving Patterns

Mastering common patterns helps you recognize solutions faster. Here are the most important ones:

---

### Pattern 1: Frequency Counter

**When to Use**: Comparing frequencies of elements, finding duplicates, anagrams.

**Key Idea**: Use a hash map to count occurrences instead of nested loops.

#### Example 1: Valid Anagram
```swift
// Problem: Check if two strings are anagrams
// "listen" and "silent" → true

// ❌ Brute Force: Sort both - O(n log n)
func isAnagram1(_ s: String, _ t: String) -> Bool {
    return s.sorted() == t.sorted()
}

// ✅ Frequency Counter - O(n) time, O(n) space
func isAnagram(_ s: String, _ t: String) -> Bool {
    guard s.count == t.count else { return false }
    
    var freq = [Character: Int]()
    
    // Count characters in s
    for char in s {
        freq[char, default: 0] += 1
    }
    
    // Subtract characters from t
    for char in t {
        guard let count = freq[char], count > 0 else {
            return false
        }
        freq[char] = count - 1
    }
    
    return true
}

print(isAnagram("listen", "silent"))  // true
print(isAnagram("hello", "world"))    // false
```

#### Example 2: First Unique Character
```swift
// Problem: Find first non-repeating character
// "leetcode" → "l"

func firstUniqChar(_ s: String) -> Character? {
    var freq = [Character: Int]()
    
    // Count all characters
    for char in s {
        freq[char, default: 0] += 1
    }
    
    // Find first with count 1
    for char in s {
        if freq[char] == 1 {
            return char
        }
    }
    
    return nil
}

print(firstUniqChar("leetcode") ?? "None")  // l
print(firstUniqChar("loveleetcode") ?? "None")  // v
```

**Pro Tip**: Always consider hash maps when you need to track frequency or existence!

---

### Pattern 2: Multiple Pointers

**When to Use**: Searching pairs in sorted arrays, removing duplicates, palindromes.

**Key Idea**: Use two or more pointers moving through the data structure.

#### Example 1: Two Sum II (Sorted Array)
```swift
// Problem: Find two numbers that sum to target in SORTED array
// [2, 7, 11, 15], target = 9 → [0, 1]

func twoSumSorted(_ numbers: [Int], _ target: Int) -> [Int] {
    var left = 0
    var right = numbers.count - 1
    
    while left < right {
        let sum = numbers[left] + numbers[right]
        
        if sum == target {
            return [left, right]
        } else if sum < target {
            left += 1  // Need larger sum
        } else {
            right -= 1  // Need smaller sum
        }
    }
    
    return []
}

print(twoSumSorted([2, 7, 11, 15], 9))  // [0, 1]
```

#### Example 2: Valid Palindrome
```swift
// Problem: Check if string is palindrome (ignoring non-alphanumeric)
// "A man, a plan, a canal: Panama" → true

func isPalindrome(_ s: String) -> Bool {
    let chars = Array(s.lowercased().filter { $0.isLetter || $0.isNumber })
    
    var left = 0
    var right = chars.count - 1
    
    while left < right {
        if chars[left] != chars[right] {
            return false
        }
        left += 1
        right -= 1
    }
    
    return true
}

print(isPalindrome("A man, a plan, a canal: Panama"))  // true
print(isPalindrome("race a car"))  // false
```

#### Example 3: Remove Duplicates (In-Place)
```swift
// Problem: Remove duplicates from sorted array in-place
// [1, 1, 2, 2, 3] → [1, 2, 3]

func removeDuplicates(_ nums: inout [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    
    var slow = 0  // Position for unique elements
    
    for fast in 1..<nums.count {
        if nums[fast] != nums[slow] {
            slow += 1
            nums[slow] = nums[fast]
        }
    }
    
    return slow + 1  // Length of unique array
}

var arr = [1, 1, 2, 2, 3]
let length = removeDuplicates(&arr)
print(Array(arr[0..<length]))  // [1, 2, 3]
```

**Pro Tip**: Two pointers work great on sorted data or when you need in-place operations!

---

### Pattern 3: Sliding Window

**When to Use**: Subarray/substring problems, finding max/min in ranges.

**Key Idea**: Maintain a window that slides through the array/string.

#### Example 1: Maximum Sum Subarray (Fixed Size)
```swift
// Problem: Find max sum of k consecutive elements
// [1, 4, 2, 10, 2, 3, 1, 0, 20], k=4 → 24

// ❌ Brute Force - O(n*k)
func maxSumBrute(_ arr: [Int], _ k: Int) -> Int {
    var maxSum = Int.min
    
    for i in 0...(arr.count - k) {
        var sum = 0
        for j in i..<(i + k) {
            sum += arr[j]
        }
        maxSum = max(maxSum, sum)
    }
    
    return maxSum
}

// ✅ Sliding Window - O(n)
func maxSum(_ arr: [Int], _ k: Int) -> Int {
    guard arr.count >= k else { return 0 }
    
    // Calculate first window
    var windowSum = arr[0..<k].reduce(0, +)
    var maxSum = windowSum
    
    // Slide the window
    for i in k..<arr.count {
        windowSum = windowSum - arr[i - k] + arr[i]
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}

let nums = [1, 4, 2, 10, 2, 3, 1, 0, 20]
print(maxSum(nums, 4))  // 24 (10+2+3+1 or 2+3+1+0 or... wait: 10+2+3+1=16... 2+10+2=14... Let me recalculate)
// Actually: [10, 2, 3, 1] = 16, [2, 3, 1, 0] = 6, [3, 1, 0, 20] = 24 ✓
```

#### Example 2: Longest Substring Without Repeating Characters
```swift
// Problem: Find length of longest substring without repeating chars
// "abcabcbb" → 3 ("abc")

func lengthOfLongestSubstring(_ s: String) -> Int {
    let chars = Array(s)
    var seen = [Character: Int]()  // char: index
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        // If char exists in window, shrink from left
        if let lastIndex = seen[char], lastIndex >= start {
            start = lastIndex + 1
        }
        
        seen[char] = end
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(lengthOfLongestSubstring("abcabcbb"))  // 3
print(lengthOfLongestSubstring("bbbbb"))     // 1
print(lengthOfLongestSubstring("pwwkew"))    // 3
```

**Pro Tip**: Sliding window is your best friend for substring/subarray problems!

---

### Pattern 4: Fast & Slow Pointers (Floyd's Algorithm)

**When to Use**: Detecting cycles, finding middle element, linked list problems.

**Key Idea**: Two pointers move at different speeds.

#### Example: Linked List Cycle
```swift
class ListNode {
    var val: Int
    var next: ListNode?
    init(_ val: Int) {
        self.val = val
        self.next = nil
    }
}

func hasCycle(_ head: ListNode?) -> Bool {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        // If they meet, there's a cycle
        if slow === fast {
            return true
        }
    }
    
    return false
}

// If cycle exists, fast will eventually catch up to slow
```

#### Example: Middle of Linked List
```swift
func middleNode(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    return slow  // When fast reaches end, slow is at middle
}
```

---

### Pattern 5: Divide & Conquer

**When to Use**: Binary search, merge sort, quick sort.

**Key Idea**: Break problem into smaller subproblems, solve recursively.

#### Example: Binary Search
```swift
func binarySearch(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2  // Avoid overflow
        
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1  // Not found
}

let sorted = [1, 3, 5, 7, 9, 11, 13]
print(binarySearch(sorted, 7))   // 3
print(binarySearch(sorted, 10))  // -1
```

---

### Pattern 6: Greedy

**When to Use**: Optimization problems where local optimal leads to global optimal.

**Key Idea**: Make the best choice at each step.

#### Example: Maximum Product of Three Numbers
```swift
// Problem: Find max product of any three numbers
// [-10, -10, 1, 3, 2] → 300 (-10 * -10 * 3)

func maximumProduct(_ nums: [Int]) -> Int {
    let sorted = nums.sorted()
    let n = sorted.count
    
    // Either: 3 largest OR 2 smallest (negative) * largest
    let product1 = sorted[n-1] * sorted[n-2] * sorted[n-3]
    let product2 = sorted[0] * sorted[1] * sorted[n-1]
    
    return max(product1, product2)
}

print(maximumProduct([1, 2, 3]))           // 6
print(maximumProduct([-10, -10, 1, 3, 2])) // 300
```
### Pattern Recognition Cheat Sheet

| Problem Type | Pattern | Hint |
|--------------|---------|------|
| Counting elements | Frequency Counter | Use hash map |
| Sorted array pairs | Two Pointers | Move from both ends |
| Subarray/substring | Sliding Window | Expand/shrink window |
| Linked list cycle | Fast & Slow | Different speeds |
| Finding in sorted | Binary Search | Divide & conquer |
| Character/digit manipulation | String/Math tricks | ASCII values, modulo |
| Optimization | Greedy | Best local choice |

<a name="chapter-4"></a>
## Chapter 4: Arrays Deep Dive

Arrays are the most fundamental data structure. Mastering them is crucial!

### Array Basics

```swift
// Declaration
var arr1: [Int] = []
var arr2 = [Int]()
var arr3: [String] = Array(repeating: "default", count: 5)

// Time Complexities
// Access:      O(1)
// Search:      O(n)
// Insert/Delete at end:   O(1) amortized
// Insert/Delete at start: O(n)
// Insert/Delete at middle: O(n)
```

### Essential Array Techniques

#### 1. Prefix Sum
**Use Case**: Quick range sum queries

```swift
// Problem: Calculate sum of elements in range [left, right] multiple times
// Array: [1, 2, 3, 4, 5]
// Query: sum(1, 3) → 2+3+4 = 9

class PrefixSum {
    var prefix: [Int]
    
    init(_ arr: [Int]) {
        prefix = [0]
        for num in arr {
            prefix.append(prefix.last! + num)
        }
        // prefix = [0, 1, 3, 6, 10, 15]
    }
    
    func rangeSum(_ left: Int, _ right: Int) -> Int {
        return prefix[right + 1] - prefix[left]
    }
}

let ps = PrefixSum([1, 2, 3, 4, 5])
print(ps.rangeSum(1, 3))  // 9
print(ps.rangeSum(0, 4))  // 15
```

#### 2. Kadane's Algorithm (Maximum Subarray)
**Use Case**: Find contiguous subarray with largest sum

```swift
// Problem: Find max sum of contiguous subarray
// [-2, 1, -3, 4, -1, 2, 1, -5, 4] → 6 (subarray [4, -1, 2, 1])

func maxSubArray(_ nums: [Int]) -> Int {
    var maxSum = nums[0]
    var currentSum = nums[0]
    
    for i in 1..<nums.count {
        // Either extend current subarray or start new one
        currentSum = max(nums[i], currentSum + nums[i])
        maxSum = max(maxSum, currentSum)
    }
    
    return maxSum
}

print(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  // 6

// To also track the subarray:
func maxSubArrayWithIndices(_ nums: [Int]) -> (sum: Int, start: Int, end: Int) {
    var maxSum = nums[0]
    var currentSum = nums[0]
    var start = 0, end = 0, tempStart = 0
    
    for i in 1..<nums.count {
        if nums[i] > currentSum + nums[i] {
            currentSum = nums[i]
            tempStart = i
        } else {
            currentSum = currentSum + nums[i]
        }
        
        if currentSum > maxSum {
            maxSum = currentSum
            start = tempStart
            end = i
        }
    }
    
    return (maxSum, start, end)
}
```

#### 3. Dutch National Flag (3-way Partitioning)
**Use Case**: Sort array with 3 distinct values

```swift
// Problem: Sort array of 0s, 1s, and 2s in-place
// [2, 0, 2, 1, 1, 0] → [0, 0, 1, 1, 2, 2]

func sortColors(_ nums: inout [Int]) {
    var low = 0      // Boundary for 0s
    var mid = 0      // Current element
    var high = nums.count - 1  // Boundary for 2s
    
    while mid <= high {
        if nums[mid] == 0 {
            nums.swapAt(low, mid)
            low += 1
            mid += 1
        } else if nums[mid] == 1 {
            mid += 1
        } else {  // nums[mid] == 2
            nums.swapAt(mid, high)
            high -= 1
            // Don't increment mid - need to check swapped element
        }
    }
}

var colors = [2, 0, 2, 1, 1, 0]
sortColors(&colors)
print(colors)  // [0, 0, 1, 1, 2, 2]
```

#### 4. Product of Array Except Self
**Use Case**: Calculate products without division

```swift
// Problem: For each index, calculate product of all other elements
// [1, 2, 3, 4] → [24, 12, 8, 6]
// Constraint: O(n) time, no division

func productExceptSelf(_ nums: [Int]) -> [Int] {
    let n = nums.count
    var result = Array(repeating: 1, count: n)
    
    // Left products
    var leftProduct = 1
    for i in 0..<n {
        result[i] = leftProduct
        leftProduct *= nums[i]
    }
    // result = [1, 1, 2, 6]
    
    // Right products
    var rightProduct = 1
    for i in (0..<n).reversed() {
        result[i] *= rightProduct
        rightProduct *= nums[i]
    }
    // result = [24, 12, 8, 6]
    
    return result
}

print(productExceptSelf([1, 2, 3, 4]))  // [24, 12, 8, 6]
```

#### 5. Rotate Array
```swift
// Problem: Rotate array to right by k steps
// [1, 2, 3, 4, 5, 6, 7], k=3 → [5, 6, 7, 1, 2, 3, 4]

func rotate(_ nums: inout [Int], _ k: Int) {
    let k = k % nums.count  // Handle k > n
    
    // Reverse entire array
    nums.reverse()
    // [7, 6, 5, 4, 3, 2, 1]
    
    // Reverse first k elements
    nums[0..<k].reverse()
    // [5, 6, 7, 4, 3, 2, 1]
    
    // Reverse remaining elements
    nums[k...].reverse()
    // [5, 6, 7, 1, 2, 3, 4]
}

var arr = [1, 2, 3, 4, 5, 6, 7]
rotate(&arr, 3)
print(arr)  // [5, 6, 7, 1, 2, 3, 4]
```

### Common Array Problems

#### Problem 1: Contains Duplicate
```swift
func containsDuplicate(_ nums: [Int]) -> Bool {
    var seen = Set<Int>()
    for num in nums {
        if seen.contains(num) {
            return true
        }
        seen.insert(num)
    }
    return false
}

// Or shorter:
func containsDuplicate2(_ nums: [Int]) -> Bool {
    return Set(nums).count != nums.count
}
```

#### Problem 2: Missing Number
```swift
// [3, 0, 1] → 2 (missing from 0...3)

// Method 1: Sum formula
func missingNumber(_ nums: [Int]) -> Int {
    let n = nums.count
    let expectedSum = n * (n + 1) / 2
    let actualSum = nums.reduce(0, +)
    return expectedSum - actualSum
}

// Method 2: XOR (handles overflow better)
func missingNumber2(_ nums: [Int]) -> Int {
    var result = nums.count
    for (i, num) in nums.enumerated() {
        result ^= i ^ num
    }
    return result
}
```

#### Problem 3: Find All Duplicates
```swift
// Find duplicates in array where 1 ≤ nums[i] ≤ n
// [4, 3, 2, 7, 8, 2, 3, 1] → [2, 3]

func findDuplicates(_ nums: inout [Int]) -> [Int] {
    var result = [Int]()
    
    // Use indices as markers
    for i in 0..<nums.count {
        let index = abs(nums[i]) - 1
        
        if nums[index] < 0 {
            // Already seen this number
            result.append(abs(nums[i]))
        } else {
            // Mark as seen by negating
            nums[index] = -nums[index]
        }
    }
    
    return result
}
```

### Array Tricks & Tips

1. **Use indices creatively**: When array values are bounded (1 to n), use them as indices
2. **In-place swapping**: Avoid extra space by clever swapping
3. **Reverse trick**: Many rotation/reordering problems solved by reversing segments
4. **Prefix/Suffix arrays**: Precompute for range queries
5. **Sentinel values**: Use min/max values to avoid edge case checks

---

**🎯 Practice Problems:**
1. Merge Sorted Arrays
2. Best Time to Buy and Sell Stock
3. Container With Most Water
4. 3Sum
5. Trapping Rain Water

<a name="chapter-5"></a>
## Chapter 5: String Manipulation

Strings are just arrays of characters, but with unique challenges and tricks!

### String Basics in Swift

```swift
// String creation
let str1 = "Hello"
let str2 = String("World")
let str3 = String(repeating: "a", count: 5)  // "aaaaa"

// Converting to Array (often needed for manipulation)
let chars = Array("Hello")  // ['H', 'e', 'l', 'l', 'o']

// String is a Collection
let s = "Swift"
print(s.count)           // 5
print(s.isEmpty)         // false
print(s.first)           // Optional('S')
print(s.last)            // Optional('t')

// Character access (slower than arrays)
let index = s.index(s.startIndex, offsetBy: 2)
print(s[index])          // 'i'

// Substring
let sub = s[s.startIndex..<index]  // "Sw"
```

### String Complexity in Swift

⚠️ **Important**: Swift strings are not simple arrays!

```swift
// Time Complexities
// Access by index:  O(n) - not O(1)!
// Length:           O(1)
// Append:           O(1) amortized
// Insert/Delete:    O(n)

// TIP: Convert to [Character] for O(1) indexing
let str = "Hello"
let chars = Array(str)  // Now O(1) access!
```

### Essential String Techniques

#### 1. Character Frequency & Anagrams

```swift
// Problem: Group anagrams together
// ["eat", "tea", "tan", "ate", "nat", "bat"]
// → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]

func groupAnagrams(_ strs: [String]) -> [[String]] {
    var groups = [String: [String]]()
    
    for str in strs {
        // Use sorted string as key
        let key = String(str.sorted())
        groups[key, default: []].append(str)
    }
    
    return Array(groups.values)
}

print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
// [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]

// Alternative: Use character frequency as key
func groupAnagrams2(_ strs: [String]) -> [[String]] {
    var groups = [[Int]: [String]]()
    
    for str in strs {
        var freq = Array(repeating: 0, count: 26)
        for char in str {
            let index = Int(char.asciiValue! - Character("a").asciiValue!)
            freq[index] += 1
        }
        groups[freq, default: []].append(str)
    }
    
    return Array(groups.values)
}
```

#### 2. Palindrome Techniques

```swift
// Check if string is palindrome (ignoring case & non-alphanumeric)
func isPalindrome(_ s: String) -> Bool {
    let cleaned = s.lowercased().filter { $0.isLetter || $0.isNumber }
    let chars = Array(cleaned)
    
    var left = 0
    var right = chars.count - 1
    
    while left < right {
        if chars[left] != chars[right] {
            return false
        }
        left += 1
        right -= 1
    }
    
    return true
}

// Valid Palindrome II: Can delete at most one character
func validPalindrome(_ s: String) -> Bool {
    let chars = Array(s)
    var left = 0
    var right = chars.count - 1
    
    while left < right {
        if chars[left] != chars[right] {
            // Try deleting either left or right character
            return isPalindrome(chars, left + 1, right) ||
                   isPalindrome(chars, left, right - 1)
        }
        left += 1
        right -= 1
    }
    
    return true
    
    func isPalindrome(_ chars: [Character], _ l: Int, _ r: Int) -> Bool {
        var left = l, right = r
        while left < right {
            if chars[left] != chars[right] {
                return false
            }
            left += 1
            right -= 1
        }
        return true
    }
}

print(validPalindrome("aba"))      // true
print(validPalindrome("abca"))     // true (delete 'c')
print(validPalindrome("abc"))      // false
```

#### 3. Longest Palindromic Substring

```swift
// Problem: Find longest palindromic substring
// "babad" → "bab" or "aba"

func longestPalindrome(_ s: String) -> String {
    guard s.count > 1 else { return s }
    
    let chars = Array(s)
    var start = 0
    var maxLength = 0
    
    // Expand around center for each possible center
    for i in 0..<chars.count {
        // Odd length palindromes (single center)
        let len1 = expandAroundCenter(chars, i, i)
        // Even length palindromes (two centers)
        let len2 = expandAroundCenter(chars, i, i + 1)
        
        let len = max(len1, len2)
        
        if len > maxLength {
            maxLength = len
            start = i - (len - 1) / 2
        }
    }
    
    return String(chars[start..<(start + maxLength)])
    
    func expandAroundCenter(_ chars: [Character], _ left: Int, _ right: Int) -> Int {
        var l = left, r = right
        while l >= 0 && r < chars.count && chars[l] == chars[r] {
            l -= 1
            r += 1
        }
        return r - l - 1
    }
}

print(longestPalindrome("babad"))   // "bab" or "aba"
print(longestPalindrome("cbbd"))    // "bb"
```

#### 4. String Reversal Variations

```swift
// Reverse entire string
func reverseString(_ s: inout [Character]) {
    var left = 0
    var right = s.count - 1
    
    while left < right {
        s.swapAt(left, right)
        left += 1
        right -= 1
    }
}

// Reverse words in a string
// "the sky is blue" → "blue is sky the"
func reverseWords(_ s: String) -> String {
    let words = s.split(separator: " ").map(String.init)
    return words.reversed().joined(separator: " ")
}

// Or without built-in split:
func reverseWords2(_ s: String) -> String {
    var chars = Array(s)
    
    // Step 1: Reverse entire string
    chars.reverse()
    // "eulb si yks eht"
    
    // Step 2: Reverse each word
    var start = 0
    for i in 0...chars.count {
        if i == chars.count || chars[i] == " " {
            reverseRange(&chars, start, i - 1)
            start = i + 1
        }
    }
    
    return String(chars)
    
    func reverseRange(_ arr: inout [Character], _ left: Int, _ right: Int) {
        var l = left, r = right
        while l < r {
            arr.swapAt(l, r)
            l += 1
            r -= 1
        }
    }
}

// Reverse vowels only
// "hello" → "holle"
func reverseVowels(_ s: String) -> String {
    var chars = Array(s)
    let vowels: Set<Character> = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
    
    var left = 0
    var right = chars.count - 1
    
    while left < right {
        while left < right && !vowels.contains(chars[left]) {
            left += 1
        }
        while left < right && !vowels.contains(chars[right]) {
            right -= 1
        }
        
        if left < right {
            chars.swapAt(left, right)
            left += 1
            right -= 1
        }
    }
    
    return String(chars)
}
```

#### 5. Substring Search (Pattern Matching)

```swift
// Find first occurrence of needle in haystack
// "hello", "ll" → 2

// Simple approach - O(n*m)
func strStr(_ haystack: String, _ needle: String) -> Int {
    guard !needle.isEmpty else { return 0 }
    guard haystack.count >= needle.count else { return -1 }
    
    let h = Array(haystack)
    let n = Array(needle)
    
    for i in 0...(h.count - n.count) {
        if h[i..<(i + n.count)] == n[0..<n.count] {
            return i
        }
    }
    
    return -1
}

// Or use Swift's built-in:
func strStr2(_ haystack: String, _ needle: String) -> Int {
    guard let range = haystack.range(of: needle) else {
        return -1
    }
    return haystack.distance(from: haystack.startIndex, to: range.lowerBound)
}
```

#### 6. String Compression

```swift
// Compress string: "aabcccccaaa" → "a2b1c5a3"
// If compressed is not shorter, return original

func compress(_ chars: inout [Character]) -> Int {
    var write = 0
    var read = 0
    
    while read < chars.count {
        let char = chars[read]
        var count = 0
        
        // Count consecutive characters
        while read < chars.count && chars[read] == char {
            read += 1
            count += 1
        }
        
        // Write character
        chars[write] = char
        write += 1
        
        // Write count if > 1
        if count > 1 {
            for digit in String(count) {
                chars[write] = digit
                write += 1
            }
        }
    }
    
    return write
}

var chars: [Character] = ["a", "a", "b", "b", "c", "c", "c"]
let newLength = compress(&chars)
print(String(chars[0..<newLength]))  // "a2b2c3"
```

### Advanced String Problems

#### Problem 1: Longest Substring Without Repeating Characters
```swift
// "abcabcbb" → 3 ("abc")
// "bbbbb" → 1 ("b")
// "pwwkew" → 3 ("wke")

func lengthOfLongestSubstring(_ s: String) -> Int {
    let chars = Array(s)
    var seen = [Character: Int]()  // char: last index
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        // If char in current window, move start past its last occurrence
        if let lastIndex = seen[char], lastIndex >= start {
            start = lastIndex + 1
        }
        
        seen[char] = end
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}
```

#### Problem 2: Minimum Window Substring
```swift
// Find smallest substring in s containing all characters from t
// s = "ADOBECODEBANC", t = "ABC" → "BANC"

func minWindow(_ s: String, _ t: String) -> String {
    guard s.count >= t.count else { return "" }
    
    let sChars = Array(s)
    var tFreq = [Character: Int]()
    
    // Count characters in t
    for char in t {
        tFreq[char, default: 0] += 1
    }
    
    var required = tFreq.count
    var formed = 0
    var windowCounts = [Character: Int]()
    
    var left = 0
    var minLen = Int.max
    var minLeft = 0
    
    for right in 0..<sChars.count {
        let char = sChars[right]
        windowCounts[char, default: 0] += 1
        
        if let tCount = tFreq[char], windowCounts[char] == tCount {
            formed += 1
        }
        
        // Try to shrink window
        while formed == required && left <= right {
            // Update result
            if right - left + 1 < minLen {
                minLen = right - left + 1
                minLeft = left
            }
            
            // Remove from left
            let leftChar = sChars[left]
            windowCounts[leftChar]! -= 1
            if let tCount = tFreq[leftChar], windowCounts[leftChar]! < tCount {
                formed -= 1
            }
            left += 1
        }
    }
    
    return minLen == Int.max ? "" : String(sChars[minLeft..<(minLeft + minLen)])
}

print(minWindow("ADOBECODEBANC", "ABC"))  // "BANC"
```

#### Problem 3: Decode String
```swift
// "3[a]2[bc]" → "aaabcbc"
// "3[a2[c]]" → "accaccacc"

func decodeString(_ s: String) -> String {
    var countStack = [Int]()
    var stringStack = [String]()
    var currentString = ""
    var currentNum = 0
    
    for char in s {
        if char.isNumber {
            currentNum = currentNum * 10 + Int(String(char))!
        } else if char == "[" {
            countStack.append(currentNum)
            stringStack.append(currentString)
            currentNum = 0
            currentString = ""
        } else if char == "]" {
            let prevString = stringStack.removeLast()
            let count = countStack.removeLast()
            currentString = prevString + String(repeating: currentString, count: count)
        } else {
            currentString.append(char)
        }
    }
    
    return currentString
}

print(decodeString("3[a]2[bc]"))      // "aaabcbc"
print(decodeString("3[a2[c]]"))       // "accaccacc"
print(decodeString("2[abc]3[cd]ef"))  // "abcabccdcdcdef"
```

### String Building Performance

```swift
// ❌ SLOW - O(n²) due to string immutability
func buildStringSlow(_ n: Int) -> String {
    var result = ""
    for i in 0..<n {
        result += "a"  // Creates new string each time!
    }
    return result
}

// ✅ FAST - O(n) using array
func buildStringFast(_ n: Int) -> String {
    var chars = [Character]()
    for i in 0..<n {
        chars.append("a")
    }
    return String(chars)
}

// ✅ ALSO FAST - Using String's methods
func buildStringFast2(_ n: Int) -> String {
    return String(repeating: "a", count: n)
}
```

### Common String Patterns

#### Pattern 1: Character Mapping
```swift
// Map characters to indices (for lowercase)
func charToIndex(_ char: Character) -> Int {
    return Int(char.asciiValue! - Character("a").asciiValue!)
}

func indexToChar(_ index: Int) -> Character {
    return Character(UnicodeScalar(index + Int(Character("a").asciiValue!))!)
}

print(charToIndex("c"))  // 2
print(indexToChar(2))    // "c"
```

#### Pattern 2: Sliding Window for Substrings
```swift
// Find all substrings of length k with k distinct characters
func kDistinctChars(_ s: String, _ k: Int) -> [String] {
    let chars = Array(s)
    var result = [String]()
    var freq = [Character: Int]()
    
    for i in 0..<chars.count {
        freq[chars[i], default: 0] += 1
        
        if i >= k {
            let leftChar = chars[i - k]
            freq[leftChar]! -= 1
            if freq[leftChar] == 0 {
                freq.removeValue(forKey: leftChar)
            }
        }
        
        if i >= k - 1 && freq.count == k {
            result.append(String(chars[(i - k + 1)...i]))
        }
    }
    
    return result
}
```

### String Tricks & Tips

1. **Convert to array**: For frequent indexing, convert to `[Character]`
2. **Character sets**: Use `Set<Character>` for vowels, special chars, etc.
3. **ASCII math**: Use ASCII values for character manipulation
4. **StringBuilder pattern**: Use array append, then convert to String
5. **Two pointers**: Great for palindromes and reversal problems
6. **Sliding window**: Perfect for substring problems
7. **Stack**: Useful for matching brackets, decode problems

<a name="chapter-6"></a>
## Chapter 6: Two Pointers Technique

The two pointers technique is one of the most powerful patterns for array/string problems!

### When to Use Two Pointers

✅ **Use when:**
- Array/string is sorted
- Need to find pairs/triplets
- Removing elements in-place
- Palindrome checking
- Partitioning problems

### Types of Two Pointers

#### Type 1: Opposite Direction (Converging)
```
[1, 2, 3, 4, 5, 6]
 ↑              ↑
 L              R
```

#### Type 2: Same Direction (Fast & Slow)
```
[1, 2, 3, 4, 5, 6]
 ↑  ↑
 S  F
```

#### Type 3: Fixed Window
```
[1, 2, 3, 4, 5, 6]
 ↑     ↑
 L     R (R - L = k)
```

---

### Pattern 1: Opposite Direction Pointers

#### Problem 1: Two Sum (Sorted Array)
```swift
// Given sorted array, find two numbers that sum to target
func twoSum(_ numbers: [Int], _ target: Int) -> [Int] {
    var left = 0
    var right = numbers.count - 1
    
    while left < right {
        let sum = numbers[left] + numbers[right]
        
        if sum == target {
            return [left, right]  // or [left + 1, right + 1] for 1-indexed
        } else if sum < target {
            left += 1  // Need larger sum
        } else {
            right -= 1  // Need smaller sum
        }
    }
    
    return []
}

print(twoSum([2, 7, 11, 15], 9))  // [0, 1]
```

#### Problem 2: Three Sum
```swift
// Find all unique triplets that sum to zero
// [-1, 0, 1, 2, -1, -4] → [[-1, -1, 2], [-1, 0, 1]]

func threeSum(_ nums: [Int]) -> [[Int]] {
    guard nums.count >= 3 else { return [] }
    
    let sorted = nums.sorted()
    var result = [[Int]]()
    
    for i in 0..<sorted.count - 2 {
        // Skip duplicates for first number
        if i > 0 && sorted[i] == sorted[i - 1] {
            continue
        }
        
        // Two sum on remaining array
        var left = i + 1
        var right = sorted.count - 1
        let target = -sorted[i]
        
        while left < right {
            let sum = sorted[left] + sorted[right]
            
            if sum == target {
                result.append([sorted[i], sorted[left], sorted[right]])
                
                // Skip duplicates
                while left < right && sorted[left] == sorted[left + 1] {
                    left += 1
                }
                while left < right && sorted[right] == sorted[right - 1] {
                    right -= 1
                }
                
                left += 1
                right -= 1
            } else if sum < target {
                left += 1
            } else {
                right -= 1
            }
        }
    }
    
    return result
}

print(threeSum([-1, 0, 1, 2, -1, -4]))  // [[-1, -1, 2], [-1, 0, 1]]
```

#### Problem 3: Container With Most Water
```swift
// Find two lines that form container with maximum area
// [1, 8, 6, 2, 5, 4, 8, 3, 7] → 49

func maxArea(_ height: [Int]) -> Int {
    var left = 0
    var right = height.count - 1
    var maxArea = 0
    
    while left < right {
        let width = right - left
        let h = min(height[left], height[right])
        let area = width * h
        
        maxArea = max(maxArea, area)
        
        // Move pointer with smaller height
        if height[left] < height[right] {
            left += 1
        } else {
            right -= 1
        }
    }
    
    return maxArea
}

print(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))  // 49
```

#### Problem 4: Trapping Rain Water
```swift
// Calculate trapped rainwater between bars
// [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1] → 6

func trap(_ height: [Int]) -> Int {
    guard height.count > 2 else { return 0 }
    
    var left = 0
    var right = height.count - 1
    var leftMax = 0
    var rightMax = 0
    var water = 0
    
    while left < right {
        if height[left] < height[right] {
            if height[left] >= leftMax {
                leftMax = height[left]
            } else {
                water += leftMax - height[left]
            }
            left += 1
        } else {
            if height[right] >= rightMax {
                rightMax = height[right]
            } else {
                water += rightMax - height[right]
            }
            right -= 1
        }
    }
    
    return water
}

print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  // 6
```

---

### Pattern 2: Same Direction (Fast & Slow)

#### Problem 1: Remove Duplicates
```swift
// Remove duplicates in-place from sorted array
// [1, 1, 2, 2, 3] → [1, 2, 3], return 3

func removeDuplicates(_ nums: inout [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    
    var slow = 0  // Position for unique elements
    
    for fast in 1..<nums.count {
        if nums[fast] != nums[slow] {
            slow += 1
            nums[slow] = nums[fast]
        }
    }
    
    return slow + 1
}

var arr = [1, 1, 2, 2, 3, 3, 3]
let len = removeDuplicates(&arr)
print(Array(arr[0..<len]))  // [1, 2, 3]
```

#### Problem 2: Move Zeroes
```swift
// Move all zeros to end while maintaining order
// [0, 1, 0, 3, 12] → [1, 3, 12, 0, 0]

func moveZeroes(_ nums: inout [Int]) {
    var slow = 0  // Position for non-zero elements
    
    // Move all non-zero elements to front
    for fast in 0..<nums.count {
        if nums[fast] != 0 {
            nums[slow] = nums[fast]
            slow += 1
        }
    }
    
    // Fill rest with zeros
    for i in slow..<nums.count {
        nums[i] = 0
    }
}

// Optimized version with swapping
func moveZeroes2(_ nums: inout [Int]) {
    var slow = 0
    
    for fast in 0..<nums.count {
        if nums[fast] != 0 {
            nums.swapAt(slow, fast)
            slow += 1
        }
    }
}

var arr2 = [0, 1, 0, 3, 12]
moveZeroes2(&arr2)
print(arr2)  // [1, 3, 12, 0, 0]
```

#### Problem 3: Remove Element
```swift
// Remove all instances of val in-place
// [3, 2, 2, 3], val = 3 → [2, 2], return 2

func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
    var slow = 0
    
    for fast in 0..<nums.count {
        if nums[fast] != val {
            nums[slow] = nums[fast]
            slow += 1
        }
    }
    
    return slow
}

var arr3 = [3, 2, 2, 3]
let newLen = removeElement(&arr3, 3)
print(Array(arr3[0..<newLen]))  // [2, 2]
```

#### Problem 4: Linked List Cycle Detection (Floyd's Algorithm)
```swift
class ListNode {
    var val: Int
    var next: ListNode?
    init(_ val: Int) {
        self.val = val
    }
}

func hasCycle(_ head: ListNode?) -> Bool {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        if slow === fast {
            return true
        }
    }
    
    return false
}

// Find cycle start
func detectCycle(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    // Find meeting point
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        if slow === fast {
            // Move slow to head, move both at same speed
            slow = head
            while slow !== fast {
                slow = slow?.next
                fast = fast?.next
            }
            return slow
        }
    }
    
    return nil
}
```

---

### Pattern 3: Partition Problems

#### Problem 1: Sort Colors (Dutch National Flag)
```swift
// Sort array of 0s, 1s, 2s in-place
func sortColors(_ nums: inout [Int]) {
    var low = 0      // Next position for 0
    var mid = 0      // Current element
    var high = nums.count - 1  // Next position for 2
    
    while mid <= high {
        switch nums[mid] {
        case 0:
            nums.swapAt(low, mid)
            low += 1
            mid += 1
        case 1:
            mid += 1
        case 2:
            nums.swapAt(mid, high)
            high -= 1
            // Don't increment mid - need to check swapped element
        default:
            break
        }
    }
}

var colors = [2, 0, 2, 1, 1, 0]
sortColors(&colors)
print(colors)  // [0, 0, 1, 1, 2, 2]
```

#### Problem 2: Partition Array
```swift
// Partition array around pivot (Quick Sort partition)
func partition(_ nums: inout [Int], _ low: Int, _ high: Int) -> Int {
    let pivot = nums[high]
    var i = low - 1
    
    for j in low..<high {
        if nums[j] <= pivot {
            i += 1
            nums.swapAt(i, j)
        }
    }
    
    nums.swapAt(i + 1, high)
    return i + 1
}

// Partition even/odd
func partitionEvenOdd(_ nums: inout [Int]) {
    var left = 0
    var right = nums.count - 1
    
    while left < right {
        while left < right && nums[left] % 2 == 0 {
            left += 1
        }
        while left < right && nums[right] % 2 == 1 {
            right -= 1
        }
        
        if left < right {
            nums.swapAt(left, right)
        }
    }
}

var mixed = [1, 2, 3, 4, 5, 6]
partitionEvenOdd(&mixed)
print(mixed)  // [6, 2, 4, 3, 5, 1] (all evens before odds)
```

---

### Advanced Two Pointer Problems

#### Problem 1: 4Sum
```swift
// Find all unique quadruplets that sum to target
func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
    guard nums.count >= 4 else { return [] }
    
    let sorted = nums.sorted()
    var result = [[Int]]()
    
    for i in 0..<sorted.count - 3 {
        if i > 0 && sorted[i] == sorted[i - 1] { continue }
        
        for j in (i + 1)..<sorted.count - 2 {
            if j > i + 1 && sorted[j] == sorted[j - 1] { continue }
            
            var left = j + 1
            var right = sorted.count - 1
            
            while left < right {
                let sum = sorted[i] + sorted[j] + sorted[left] + sorted[right]
                
                if sum == target {
                    result.append([sorted[i], sorted[j], sorted[left], sorted[right]])
                    
                    while left < right && sorted[left] == sorted[left + 1] {
                        left += 1
                    }
                    while left < right && sorted[right] == sorted[right - 1] {
                        right -= 1
                    }
                    
                    left += 1
                    right -= 1
                } else if sum < target {
                    left += 1
                } else {
                    right -= 1
                }
            }
        }
    }
    
    return result
}
```

#### Problem 2: Subarray Product Less Than K
```swift
// Count subarrays where product < k
// [10, 5, 2, 6], k = 100 → 8

func numSubarrayProductLessThanK(_ nums: [Int], _ k: Int) -> Int {
    guard k > 1 else { return 0 }
    
    var product = 1
    var count = 0
    var left = 0
    
    for right in 0..<nums.count {
        product *= nums[right]
        
        while product >= k {
            product /= nums[left]
            left += 1
        }
        
        // All subarrays ending at right
        count += right - left + 1
    }
    
    return count
}

print(numSubarrayProductLessThanK([10, 5, 2, 6], 100))  // 8
```

### Two Pointers Cheat Sheet

| Problem Type | Pointer Setup | Movement Rule |
|--------------|---------------|---------------|
| Two Sum (sorted) | Opposite ends | Sum comparison |
| Palindrome | Opposite ends | Character comparison |
| Remove duplicates | Same direction | Skip equals |
| Move zeros | Same direction | Skip zeros |
| Partition | Opposite ends | Value comparison |
| Cycle detection | Same direction | 1x vs 2x speed |
| Container/water | Opposite ends | Move smaller height |

### Tips & Tricks

1. **Sorted array?** → Consider opposite direction pointers
2. **In-place modification?** → Consider fast & slow pointers
3. **Finding pairs?** → Opposite direction after sorting
4. **Partitioning?** → Opposite direction with condition
5. **Cycle detection?** → Fast & slow pointers
6. **Always check bounds** before dereferencing pointers!
7. **Skip duplicates** in sorted arrays for unique results

**🎯 Practice Problems:**
1. Valid Triangle Number
2. Intersection of Two Arrays II
3. Squares of Sorted Array
4. Backspace String Compare
5. Longest Mountain in Array

<a name="chapter-7"></a>
## Chapter 7: Sliding Window Pattern

The sliding window pattern is incredibly powerful for substring/subarray problems. Master this, and you'll solve dozens of problems easily!

### What is Sliding Window?

A technique where you maintain a **window** (subarray/substring) that slides through the data structure.

```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
       [-------]              Window of size 3
          [-------]           Slide right
             [-------]        Slide right
```

### When to Use Sliding Window?

✅ **Use when:**
- Finding subarrays/substrings with specific property
- Maximum/minimum of size k
- Longest/shortest substring with condition
- Problems involving contiguous sequences

### Types of Sliding Windows

#### Type 1: Fixed-Size Window
Window size is constant (k elements)

#### Type 2: Dynamic Window
Window size varies based on conditions

---

### Pattern 1: Fixed-Size Window

#### Problem 1: Maximum Sum of Subarray (Size K)
```swift
// Find maximum sum of k consecutive elements
// [2, 1, 5, 1, 3, 2], k = 3 → 9 (5+1+3)

// ❌ Brute Force - O(n*k)
func maxSumBrute(_ nums: [Int], _ k: Int) -> Int {
    var maxSum = Int.min
    
    for i in 0...(nums.count - k) {
        var sum = 0
        for j in i..<(i + k) {
            sum += nums[j]
        }
        maxSum = max(maxSum, sum)
    }
    
    return maxSum
}

// ✅ Sliding Window - O(n)
func maxSum(_ nums: [Int], _ k: Int) -> Int {
    guard nums.count >= k else { return 0 }
    
    // Calculate first window
    var windowSum = 0
    for i in 0..<k {
        windowSum += nums[i]
    }
    
    var maxSum = windowSum
    
    // Slide the window
    for i in k..<nums.count {
        windowSum = windowSum - nums[i - k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}

print(maxSum([2, 1, 5, 1, 3, 2], 3))  // 9
```

#### Problem 2: Average of Subarrays
```swift
// Find average of all subarrays of size k
// [1, 3, 2, 6, -1, 4, 1, 8, 2], k = 5

func findAverages(_ arr: [Int], _ k: Int) -> [Double] {
    var result = [Double]()
    var windowSum = 0.0
    
    for i in 0..<arr.count {
        windowSum += Double(arr[i])
        
        if i >= k - 1 {
            result.append(windowSum / Double(k))
            windowSum -= Double(arr[i - k + 1])
        }
    }
    
    return result
}

print(findAverages([1, 3, 2, 6, -1, 4, 1, 8, 2], 5))
// [2.2, 2.8, 2.4, 3.6, 2.8]
```

#### Problem 3: Maximum of All Subarrays (Size K)
```swift
// Find maximum element in each window of size k
// [1, 3, -1, -3, 5, 3, 6, 7], k = 3 → [3, 3, 5, 5, 6, 7]

// Using deque (double-ended queue) for O(n) solution
func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    guard !nums.isEmpty && k > 0 else { return [] }
    
    var result = [Int]()
    var deque = [Int]()  // Stores indices
    
    for i in 0..<nums.count {
        // Remove elements outside window
        if !deque.isEmpty && deque.first! <= i - k {
            deque.removeFirst()
        }
        
        // Remove smaller elements (they'll never be max)
        while !deque.isEmpty && nums[deque.last!] < nums[i] {
            deque.removeLast()
        }
        
        deque.append(i)
        
        // Add to result when window is complete
        if i >= k - 1 {
            result.append(nums[deque.first!])
        }
    }
    
    return result
}

print(maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
// [3, 3, 5, 5, 6, 7]
```

#### Problem 4: First Negative in Every Window
```swift
// Find first negative number in each window of size k
// [12, -1, -7, 8, -15, 30, 16, 28], k = 3
// → [-1, -1, -7, -15, -15, 0]

func firstNegative(_ arr: [Int], _ k: Int) -> [Int] {
    var result = [Int]()
    var negatives = [Int]()  // Queue of negative indices
    
    for i in 0..<arr.count {
        // Add negative numbers to queue
        if arr[i] < 0 {
            negatives.append(i)
        }
        
        // Remove elements outside window
        while !negatives.isEmpty && negatives.first! <= i - k {
            negatives.removeFirst()
        }
        
        // Add result for this window
        if i >= k - 1 {
            if negatives.isEmpty {
                result.append(0)
            } else {
                result.append(arr[negatives.first!])
            }
        }
    }
    
    return result
}

print(firstNegative([12, -1, -7, 8, -15, 30, 16, 28], 3))
// [-1, -1, -7, -15, -15, 0]
```

---

### Pattern 2: Dynamic Window (Expand/Shrink)

#### Problem 1: Longest Substring Without Repeating Characters
```swift
// "abcabcbb" → 3 ("abc")
// "bbbbb" → 1 ("b")
// "pwwkew" → 3 ("wke")

func lengthOfLongestSubstring(_ s: String) -> Int {
    let chars = Array(s)
    var seen = [Character: Int]()  // char: index
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        // If char exists in window, move start
        if let lastIndex = seen[char], lastIndex >= start {
            start = lastIndex + 1
        }
        
        seen[char] = end
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(lengthOfLongestSubstring("abcabcbb"))  // 3
print(lengthOfLongestSubstring("pwwkew"))    // 3
```

#### Problem 2: Minimum Size Subarray Sum
```swift
// Find minimum length subarray with sum ≥ target
// [2, 3, 1, 2, 4, 3], target = 7 → 2 ([4, 3])

func minSubArrayLen(_ target: Int, _ nums: [Int]) -> Int {
    var minLength = Int.max
    var sum = 0
    var start = 0
    
    for end in 0..<nums.count {
        sum += nums[end]
        
        // Shrink window while sum >= target
        while sum >= target {
            minLength = min(minLength, end - start + 1)
            sum -= nums[start]
            start += 1
        }
    }
    
    return minLength == Int.max ? 0 : minLength
}

print(minSubArrayLen(7, [2, 3, 1, 2, 4, 3]))  // 2
```

#### Problem 3: Longest Substring with At Most K Distinct Characters
```swift
// Find longest substring with at most k distinct characters
// "eceba", k = 2 → 3 ("ece")

func lengthOfLongestSubstringKDistinct(_ s: String, _ k: Int) -> Int {
    guard k > 0 else { return 0 }
    
    let chars = Array(s)
    var freq = [Character: Int]()
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        freq[char, default: 0] += 1
        
        // Shrink window if more than k distinct chars
        while freq.count > k {
            let leftChar = chars[start]
            freq[leftChar]! -= 1
            if freq[leftChar] == 0 {
                freq.removeValue(forKey: leftChar)
            }
            start += 1
        }
        
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(lengthOfLongestSubstringKDistinct("eceba", 2))  // 3
```

#### Problem 4: Longest Repeating Character Replacement
```swift
// Replace at most k characters to get longest repeating substring
// "AABABBA", k = 1 → 4 ("AABA" or "ABBB")

func characterReplacement(_ s: String, _ k: Int) -> Int {
    let chars = Array(s)
    var freq = [Character: Int]()
    var maxLength = 0
    var maxFreq = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        freq[char, default: 0] += 1
        maxFreq = max(maxFreq, freq[char]!)
        
        // Window size - most frequent char = replacements needed
        let windowSize = end - start + 1
        
        if windowSize - maxFreq > k {
            // Too many replacements needed, shrink window
            let leftChar = chars[start]
            freq[leftChar]! -= 1
            start += 1
        }
        
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(characterReplacement("AABABBA", 1))  // 4
```

#### Problem 5: Permutation in String
```swift
// Check if s2 contains permutation of s1
// s1 = "ab", s2 = "eidbaooo" → true

func checkInclusion(_ s1: String, _ s2: String) -> Bool {
    guard s1.count <= s2.count else { return false }
    
    let s1Chars = Array(s1)
    let s2Chars = Array(s2)
    var s1Freq = [Character: Int]()
    var windowFreq = [Character: Int]()
    
    // Count s1 characters
    for char in s1Chars {
        s1Freq[char, default: 0] += 1
    }
    
    // Sliding window
    for i in 0..<s2Chars.count {
        // Add character to window
        windowFreq[s2Chars[i], default: 0] += 1
        
        // Remove character from window if beyond s1 length
        if i >= s1Chars.count {
            let leftChar = s2Chars[i - s1Chars.count]
            windowFreq[leftChar]! -= 1
            if windowFreq[leftChar] == 0 {
                windowFreq.removeValue(forKey: leftChar)
            }
        }
        
        // Check if frequencies match
        if windowFreq == s1Freq {
            return true
        }
    }
    
    return false
}

print(checkInclusion("ab", "eidbaooo"))  // true
```

---

### Pattern 3: Variable Window with Conditions

#### Problem 1: Fruit Into Baskets
```swift
// Pick maximum fruits with at most 2 types
// [1, 2, 1, 2, 3] → 4 ([1, 2, 1, 2])

func totalFruit(_ fruits: [Int]) -> Int {
    var freq = [Int: Int]()
    var maxFruits = 0
    var start = 0
    
    for (end, fruit) in fruits.enumerated() {
        freq[fruit, default: 0] += 1
        
        // More than 2 types, shrink window
        while freq.count > 2 {
            let leftFruit = fruits[start]
            freq[leftFruit]! -= 1
            if freq[leftFruit] == 0 {
                freq.removeValue(forKey: leftFruit)
            }
            start += 1
        }
        
        maxFruits = max(maxFruits, end - start + 1)
    }
    
    return maxFruits
}

print(totalFruit([1, 2, 1, 2, 3]))  // 4
```

#### Problem 2: Longest Substring with At Most Two Distinct Characters
```swift
func lengthOfLongestSubstringTwoDistinct(_ s: String) -> Int {
    let chars = Array(s)
    var freq = [Character: Int]()
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        freq[char, default: 0] += 1
        
        while freq.count > 2 {
            let leftChar = chars[start]
            freq[leftChar]! -= 1
            if freq[leftChar] == 0 {
                freq.removeValue(forKey: leftChar)
            }
            start += 1
        }
        
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(lengthOfLongestSubstringTwoDistinct("eceba"))  // 3
```

#### Problem 3: Subarrays with K Different Integers
```swift
// Count subarrays with exactly k different integers
// [1, 2, 1, 2, 3], k = 2 → 7

func subarraysWithKDistinct(_ nums: [Int], _ k: Int) -> Int {
    // Exactly k = at most k - at most (k-1)
    return atMostK(nums, k) - atMostK(nums, k - 1)
    
    func atMostK(_ nums: [Int], _ k: Int) -> Int {
        var freq = [Int: Int]()
        var count = 0
        var start = 0
        
        for (end, num) in nums.enumerated() {
            freq[num, default: 0] += 1
            
            while freq.count > k {
                let leftNum = nums[start]
                freq[leftNum]! -= 1
                if freq[leftNum] == 0 {
                    freq.removeValue(forKey: leftNum)
                }
                start += 1
            }
            
            // All subarrays ending at end
            count += end - start + 1
        }
        
        return count
    }
}

print(subarraysWithKDistinct([1, 2, 1, 2, 3], 2))  // 7
```

---

### Advanced Sliding Window Problems

#### Problem 1: Minimum Window Substring
```swift
// Find smallest substring in s containing all characters from t
// s = "ADOBECODEBANC", t = "ABC" → "BANC"

func minWindow(_ s: String, _ t: String) -> String {
    guard s.count >= t.count else { return "" }
    
    let sChars = Array(s)
    var tFreq = [Character: Int]()
    var windowFreq = [Character: Int]()
    
    for char in t {
        tFreq[char, default: 0] += 1
    }
    
    var required = tFreq.count
    var formed = 0
    var minLen = Int.max
    var minLeft = 0
    var start = 0
    
    for (end, char) in sChars.enumerated() {
        windowFreq[char, default: 0] += 1
        
        if let tCount = tFreq[char], windowFreq[char] == tCount {
            formed += 1
        }
        
        // Try to shrink window
        while formed == required {
            // Update result
            if end - start + 1 < minLen {
                minLen = end - start + 1
                minLeft = start
            }
            
            let leftChar = sChars[start]
            windowFreq[leftChar]! -= 1
            if let tCount = tFreq[leftChar], windowFreq[leftChar]! < tCount {
                formed -= 1
            }
            start += 1
        }
    }
    
    return minLen == Int.max ? "" : String(sChars[minLeft..<(minLeft + minLen)])
}

print(minWindow("ADOBECODEBANC", "ABC"))  // "BANC"
```

#### Problem 2: Max Consecutive Ones III
```swift
// Maximum consecutive 1s if you can flip at most k zeros
// [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], k = 2 → 6

func longestOnes(_ nums: [Int], _ k: Int) -> Int {
    var maxLength = 0
    var zeros = 0
    var start = 0
    
    for (end, num) in nums.enumerated() {
        if num == 0 {
            zeros += 1
        }
        
        // Too many zeros, shrink window
        while zeros > k {
            if nums[start] == 0 {
                zeros -= 1
            }
            start += 1
        }
        
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2))  // 6
```

---

### Sliding Window Templates

#### Template 1: Fixed Size Window
```swift
func fixedWindowTemplate(_ arr: [Int], _ k: Int) -> Int {
    var windowValue = 0  // sum, product, etc.
    var result = 0
    
    // Build first window
    for i in 0..<k {
        // Add arr[i] to window
    }
    
    result = windowValue
    
    // Slide window
    for i in k..<arr.count {
        // Remove arr[i-k] from window
        // Add arr[i] to window
        // Update result
    }
    
    return result
}
```

#### Template 2: Dynamic Window
```swift
func dynamicWindowTemplate(_ arr: [Int]) -> Int {
    var start = 0
    var result = 0
    var windowState = [Int: Int]()  // Track window state
    
    for (end, element) in arr.enumerated() {
        // Add element to window
        
        // Shrink window while condition not met
        while !conditionMet() {
            // Remove arr[start] from window
            start += 1
        }
        
        // Update result with current window
        result = max(result, end - start + 1)
    }
    
    return result
    
    func conditionMet() -> Bool {
        // Check if window meets condition
        return true
    }
}
```

### Sliding Window Patterns Recognition

| Problem Keywords | Pattern | Template |
|------------------|---------|----------|
| "size k" | Fixed window | Build first, then slide |
| "at most k" | Dynamic window | Expand + shrink |
| "exactly k" | at_most(k) - at_most(k-1) | Two dynamic windows |
| "minimum/maximum substring" | Dynamic window | Shrink when valid |
| "contains all" | Dynamic + counter | Track required chars |
| "consecutive" | Track runs | Count consecutive elements |

### Tips & Tricks

1. **Fixed window**: Calculate first window, then add right & remove left
2. **Dynamic window**: Expand with right, shrink with left
3. **Use hash map**: Track frequencies in current window
4. **Deque for min/max**: Maintain monotonic deque for O(n) solution
5. **Count subarrays**: Each position contributes (end - start + 1) subarrays
6. **"Exactly k" trick**: atMost(k) - atMost(k-1)
7. **Template first**: Write template, then fill in logic

---

**🎯 Practice Problems:**
1. Longest Turbulent Subarray
2. Get Equal Substrings Within Budget
3. Grumpy Bookstore Owner
4. Replace the Substring for Balanced String
5. Count Number of Nice Subarrays

<a name="chapter-8"></a>
## Chapter 8: Singly Linked Lists

Linked lists are fundamental data structures. Unlike arrays, they don't have random access, but excel at insertions and deletions!

### Linked List Basics

```swift
// Node definition
class ListNode {
    var val: Int
    var next: ListNode?
    
    init(_ val: Int) {
        self.val = val
        self.next = nil
    }
}

// Creating a linked list: 1 -> 2 -> 3
let head = ListNode(1)
head.next = ListNode(2)
head.next?.next = ListNode(3)
```

### Time Complexities

| Operation | Time | Notes |
|-----------|------|-------|
| Access | O(n) | Must traverse from head |
| Search | O(n) | Linear search only |
| Insert at head | O(1) | Just update pointers |
| Insert at tail | O(n) | Need to find tail |
| Delete at head | O(1) | Update head pointer |
| Delete elsewhere | O(n) | Need to find node |

### Essential Operations

#### 1. Traversal & Printing
```swift
func printList(_ head: ListNode?) {
    var current = head
    var values = [String]()
    
    while current != nil {
        values.append("\(current!.val)")
        current = current?.next
    }
    
    print(values.joined(separator: " -> "))
}

// Print with arrow
func printListArrow(_ head: ListNode?) {
    var current = head
    while current != nil {
        print(current!.val, terminator: "")
        if current?.next != nil {
            print(" -> ", terminator: "")
        }
        current = current?.next
    }
    print()
}
```

#### 2. Insert at Beginning
```swift
func insertAtHead(_ head: ListNode?, _ val: Int) -> ListNode {
    let newNode = ListNode(val)
    newNode.next = head
    return newNode
}

var list = ListNode(2)
list = insertAtHead(list, 1)  // 1 -> 2
```

#### 3. Insert at End
```swift
func insertAtTail(_ head: ListNode?, _ val: Int) -> ListNode {
    let newNode = ListNode(val)
    
    guard let head = head else {
        return newNode
    }
    
    var current = head
    while current.next != nil {
        current = current.next!
    }
    current.next = newNode
    
    return head
}
```

#### 4. Insert at Position
```swift
func insertAt(_ head: ListNode?, _ position: Int, _ val: Int) -> ListNode? {
    let newNode = ListNode(val)
    
    // Insert at head
    if position == 0 {
        newNode.next = head
        return newNode
    }
    
    var current = head
    var index = 0
    
    // Find position
    while current != nil && index < position - 1 {
        current = current?.next
        index += 1
    }
    
    // Insert
    if current != nil {
        newNode.next = current?.next
        current?.next = newNode
    }
    
    return head
}
```

#### 5. Delete Node by Value
```swift
func deleteNode(_ head: ListNode?, _ val: Int) -> ListNode? {
    // Handle head deletion
    var head = head
    while head != nil && head!.val == val {
        head = head!.next
    }
    
    var current = head
    while current?.next != nil {
        if current!.next!.val == val {
            current!.next = current!.next!.next
        } else {
            current = current?.next
        }
    }
    
    return head
}
```

#### 6. Length of List
```swift
func length(_ head: ListNode?) -> Int {
    var count = 0
    var current = head
    
    while current != nil {
        count += 1
        current = current?.next
    }
    
    return count
}

// Recursive version
func lengthRecursive(_ head: ListNode?) -> Int {
    guard let head = head else { return 0 }
    return 1 + lengthRecursive(head.next)
}
```

---

### Classic Linked List Problems

#### Problem 1: Reverse Linked List
```swift
// Reverse: 1 -> 2 -> 3 -> 4 → 4 -> 3 -> 2 -> 1

// Iterative - O(n) time, O(1) space
func reverseList(_ head: ListNode?) -> ListNode? {
    var prev: ListNode? = nil
    var current = head
    
    while current != nil {
        let next = current?.next
        current?.next = prev
        prev = current
        current = next
    }
    
    return prev
}

// Recursive - O(n) time, O(n) space (call stack)
func reverseListRecursive(_ head: ListNode?) -> ListNode? {
    guard let head = head, let next = head.next else {
        return head
    }
    
    let newHead = reverseListRecursive(next)
    next.next = head
    head.next = nil
    
    return newHead
}
```

#### Problem 2: Middle of Linked List
```swift
// Find middle node (if even length, return second middle)
// [1, 2, 3, 4, 5] → 3
// [1, 2, 3, 4] → 3

func middleNode(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    return slow
}
```

#### Problem 3: Detect Cycle
```swift
// Floyd's Cycle Detection Algorithm

func hasCycle(_ head: ListNode?) -> Bool {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        if slow === fast {
            return true
        }
    }
    
    return false
}

// Find where cycle begins
func detectCycle(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    // Find meeting point
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        if slow === fast {
            // Move slow to head, advance both at same speed
            slow = head
            while slow !== fast {
                slow = slow?.next
                fast = fast?.next
            }
            return slow
        }
    }
    
    return nil
}
```

#### Problem 4: Merge Two Sorted Lists
```swift
// Merge two sorted lists into one sorted list
// l1: 1 -> 2 -> 4
// l2: 1 -> 3 -> 4
// Result: 1 -> 1 -> 2 -> 3 -> 4 -> 4

func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    var current = dummy
    var l1 = l1
    var l2 = l2
    
    while l1 != nil && l2 != nil {
        if l1!.val <= l2!.val {
            current.next = l1
            l1 = l1?.next
        } else {
            current.next = l2
            l2 = l2?.next
        }
        current = current.next!
    }
    
    // Append remaining nodes
    current.next = l1 ?? l2
    
    return dummy.next
}

// Recursive version
func mergeTwoListsRecursive(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    guard let l1 = l1 else { return l2 }
    guard let l2 = l2 else { return l1 }
    
    if l1.val <= l2.val {
        l1.next = mergeTwoListsRecursive(l1.next, l2)
        return l1
    } else {
        l2.next = mergeTwoListsRecursive(l1, l2.next)
        return l2
    }
}
```

#### Problem 5: Remove Nth Node From End
```swift
// Remove nth node from end
// [1, 2, 3, 4, 5], n = 2 → [1, 2, 3, 5]

func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
    let dummy = ListNode(0)
    dummy.next = head
    
    var fast: ListNode? = dummy
    var slow: ListNode? = dummy
    
    // Move fast n steps ahead
    for _ in 0..<n {
        fast = fast?.next
    }
    
    // Move both until fast reaches end
    while fast?.next != nil {
        fast = fast?.next
        slow = slow?.next
    }
    
    // Remove node
    slow?.next = slow?.next?.next
    
    return dummy.next
}
```

#### Problem 6: Palindrome Linked List
```swift
// Check if linked list is palindrome
// [1, 2, 2, 1] → true

func isPalindrome(_ head: ListNode?) -> Bool {
    // Find middle
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    // Reverse second half
    var prev: ListNode? = nil
    while slow != nil {
        let next = slow?.next
        slow?.next = prev
        prev = slow
        slow = next
    }
    
    // Compare both halves
    var left = head
    var right = prev
    
    while right != nil {
        if left!.val != right!.val {
            return false
        }
        left = left?.next
        right = right?.next
    }
    
    return true
}
```

#### Problem 7: Intersection of Two Linked Lists
```swift
// Find node where two lists intersect
//   a1 -> a2
//             \
//              c1 -> c2 -> c3
//             /
//   b1 -> b2 -> b3

func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
    var a = headA
    var b = headB
    
    // They'll meet at intersection or both become nil
    while a !== b {
        a = a == nil ? headB : a?.next
        b = b == nil ? headA : b?.next
    }
    
    return a
}
```

#### Problem 8: Remove Duplicates from Sorted List
```swift
// Remove duplicates: 1 -> 1 -> 2 -> 3 -> 3 → 1 -> 2 -> 3

func deleteDuplicates(_ head: ListNode?) -> ListNode? {
    var current = head
    
    while current != nil && current?.next != nil {
        if current!.val == current!.next!.val {
            current!.next = current!.next!.next
        } else {
            current = current?.next
        }
    }
    
    return head
}

// Remove all duplicates (keep only unique)
// 1 -> 1 -> 2 -> 3 -> 3 → 2
func deleteDuplicates2(_ head: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    dummy.next = head
    var prev = dummy
    var current = head
    
    while current != nil {
        // Skip all duplicates
        if current?.next != nil && current!.val == current!.next!.val {
            while current?.next != nil && current!.val == current!.next!.val {
                current = current?.next
            }
            prev.next = current?.next
        } else {
            prev = prev.next!
        }
        current = current?.next
    }
    
    return dummy.next
}
```

---

### Advanced Techniques

#### Technique 1: Dummy Node
Makes edge cases easier (especially for head manipulation)

```swift
func exampleWithDummy(_ head: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    dummy.next = head
    
    // Work with dummy.next
    // ...
    
    return dummy.next
}
```

#### Technique 2: Fast & Slow Pointers
Used for finding middle, detecting cycles, finding nth from end

```swift
// Move fast 2x speed, slow 1x speed
var slow = head
var fast = head

while fast != nil && fast?.next != nil {
    slow = slow?.next
    fast = fast?.next?.next
}
// slow is now at middle
```

#### Technique 3: Runner Technique
Keep two pointers at different speeds or positions

```swift
// Fast pointer n steps ahead
var fast = head
for _ in 0..<n {
    fast = fast?.next
}

var slow = head
while fast != nil {
    slow = slow?.next
    fast = fast?.next
}
```

---

### Common Patterns & Tips

1. **Dummy head**: Use when head might change
2. **Two pointers**: Fast & slow for cycles, middle, nth from end
3. **Reversal**: Useful in many problems (palindrome, reorder)
4. **Edge cases**: Always check for nil, single node, two nodes
5. **Draw it out**: Visualize pointer movements on paper
6. **Recursive thinking**: Many problems have elegant recursive solutions
7. **Previous pointer**: Keep track of previous node for deletions

### Edge Cases Checklist

- [ ] Empty list (head == nil)
- [ ] Single node
- [ ] Two nodes
- [ ] Operations on head
- [ ] Operations on tail
- [ ] Cycle vs no cycle

**🎯 Practice Problems:**
1. Odd Even Linked List
2. Add Two Numbers (linked lists)
3. Flatten a Multilevel Doubly Linked List
4. Copy List with Random Pointer
5. LRU Cache (uses doubly linked list)

<a name="chapter-9"></a>
## Chapter 9: Doubly Linked Lists

Doubly linked lists have pointers in both directions, making bidirectional traversal possible!

### Doubly Linked List Basics

```swift
// Node definition
class DoublyListNode {
    var val: Int
    var prev: DoublyListNode?
    var next: DoublyListNode?
    
    init(_ val: Int) {
        self.val = val
        self.prev = nil
        self.next = nil
    }
}

// Creating: 1 <-> 2 <-> 3
let head = DoublyListNode(1)
let node2 = DoublyListNode(2)
let node3 = DoublyListNode(3)

head.next = node2
node2.prev = head
node2.next = node3
node3.prev = node2
```

### Advantages vs Singly Linked Lists

| Feature | Singly | Doubly |
|---------|--------|--------|
| Traverse forward | ✅ | ✅ |
| Traverse backward | ❌ | ✅ |
| Delete node (given pointer) | O(n) | O(1) |
| Memory | Less | More (extra pointer) |
| Complexity | Simpler | More complex |

### Essential Operations

#### 1. Insert at Head
```swift
func insertAtHead(_ head: DoublyListNode?, _ val: Int) -> DoublyListNode {
    let newNode = DoublyListNode(val)
    
    if let head = head {
        newNode.next = head
        head.prev = newNode
    }
    
    return newNode
}
```

#### 2. Insert at Tail
```swift
func insertAtTail(_ head: DoublyListNode?, _ val: Int) -> DoublyListNode {
    let newNode = DoublyListNode(val)
    
    guard let head = head else {
        return newNode
    }
    
    var current = head
    while current.next != nil {
        current = current.next!
    }
    
    current.next = newNode
    newNode.prev = current
    
    return head
}
```

#### 3. Insert After Node
```swift
func insertAfter(_ node: DoublyListNode, _ val: Int) {
    let newNode = DoublyListNode(val)
    
    newNode.next = node.next
    newNode.prev = node
    
    if let next = node.next {
        next.prev = newNode
    }
    
    node.next = newNode
}
```

#### 4. Delete Node (Given Pointer)
```swift
// O(1) deletion - advantage of doubly linked list!
func deleteNode(_ node: DoublyListNode) {
    if let prev = node.prev {
        prev.next = node.next
    }
    
    if let next = node.next {
        next.prev = node.prev
    }
    
    node.prev = nil
    node.next = nil
}
```

#### 5. Traverse Forward & Backward
```swift
func printForward(_ head: DoublyListNode?) {
    var current = head
    var values = [String]()
    
    while current != nil {
        values.append("\(current!.val)")
        current = current?.next
    }
    
    print(values.joined(separator: " <-> "))
}

func printBackward(_ tail: DoublyListNode?) {
    var current = tail
    var values = [String]()
    
    while current != nil {
        values.append("\(current!.val)")
        current = current?.prev
    }
    
    print(values.joined(separator: " <-> "))
}

// Get tail from head
func getTail(_ head: DoublyListNode?) -> DoublyListNode? {
    var current = head
    while current?.next != nil {
        current = current?.next
    }
    return current
}
```

#### 6. Reverse Doubly Linked List
```swift
func reverseDoublyList(_ head: DoublyListNode?) -> DoublyListNode? {
    var current = head
    var newHead: DoublyListNode? = nil
    
    while current != nil {
        // Swap next and prev
        let temp = current?.next
        current?.next = current?.prev
        current?.prev = temp
        
        newHead = current
        current = temp
    }
    
    return newHead
}
```

---

### Implementing a Doubly Linked List Class

```swift
class DoublyLinkedList {
    private var head: DoublyListNode?
    private var tail: DoublyListNode?
    private(set) var count: Int = 0
    
    var isEmpty: Bool {
        return head == nil
    }
    
    var first: DoublyListNode? {
        return head
    }
    
    var last: DoublyListNode? {
        return tail
    }
    
    // Append to end
    func append(_ value: Int) {
        let newNode = DoublyListNode(value)
        
        if let tail = tail {
            tail.next = newNode
            newNode.prev = tail
            self.tail = newNode
        } else {
            head = newNode
            tail = newNode
        }
        
        count += 1
    }
    
    // Prepend to beginning
    func prepend(_ value: Int) {
        let newNode = DoublyListNode(value)
        
        if let head = head {
            head.prev = newNode
            newNode.next = head
            self.head = newNode
        } else {
            head = newNode
            tail = newNode
        }
        
        count += 1
    }
    
    // Insert at index
    func insert(_ value: Int, at index: Int) {
        guard index >= 0 && index <= count else { return }
        
        if index == 0 {
            prepend(value)
            return
        }
        
        if index == count {
            append(value)
            return
        }
        
        let newNode = DoublyListNode(value)
        var current = head
        
        for _ in 0..<index - 1 {
            current = current?.next
        }
        
        newNode.next = current?.next
        newNode.prev = current
        current?.next?.prev = newNode
        current?.next = newNode
        
        count += 1
    }
    
    // Remove at index
    func remove(at index: Int) -> Int? {
        guard index >= 0 && index < count else { return nil }
        
        var nodeToRemove: DoublyListNode?
        
        if index == 0 {
            nodeToRemove = head
            head = head?.next
            head?.prev = nil
            if head == nil {
                tail = nil
            }
        } else if index == count - 1 {
            nodeToRemove = tail
            tail = tail?.prev
            tail?.next = nil
        } else {
            var current = head
            for _ in 0..<index {
                current = current?.next
            }
            nodeToRemove = current
            current?.prev?.next = current?.next
            current?.next?.prev = current?.prev
        }
        
        count -= 1
        return nodeToRemove?.val
    }
    
    // Remove first
    func removeFirst() -> Int? {
        return remove(at: 0)
    }
    
    // Remove last
    func removeLast() -> Int? {
        guard count > 0 else { return nil }
        return remove(at: count - 1)
    }
    
    // Get value at index
    func value(at index: Int) -> Int? {
        guard index >= 0 && index < count else { return nil }
        
        var current: DoublyListNode?
        
        // Optimize: traverse from closer end
        if index < count / 2 {
            current = head
            for _ in 0..<index {
                current = current?.next
            }
        } else {
            current = tail
            for _ in 0..<(count - index - 1) {
                current = current?.prev
            }
        }
        
        return current?.val
    }
    
    // Print list
    func printList() {
        var current = head
        var values = [String]()
        
        while current != nil {
            values.append("\(current!.val)")
            current = current?.next
        }
        
        print(values.joined(separator: " <-> "))
    }
}

// Usage
let list = DoublyLinkedList()
list.append(1)
list.append(2)
list.append(3)
list.prepend(0)
list.printList()  // 0 <-> 1 <-> 2 <-> 3

list.insert(99, at: 2)
list.printList()  // 0 <-> 1 <-> 99 <-> 2 <-> 3

list.remove(at: 2)
list.printList()  // 0 <-> 1 <-> 2 <-> 3
```

---

### LRU Cache Implementation

The most famous use of doubly linked lists!

```swift
// LRU Cache - Least Recently Used Cache
// Combines hash map + doubly linked list for O(1) operations

class LRUCache {
    class Node {
        var key: Int
        var value: Int
        var prev: Node?
        var next: Node?
        
        init(_ key: Int, _ value: Int) {
            self.key = key
            self.value = value
        }
    }
    
    private var capacity: Int
    private var cache: [Int: Node] = [:]
    private var head: Node  // Most recently used
    private var tail: Node  // Least recently used
    
    init(_ capacity: Int) {
        self.capacity = capacity
        
        // Dummy head and tail
        head = Node(0, 0)
        tail = Node(0, 0)
        head.next = tail
        tail.prev = head
    }
    
    func get(_ key: Int) -> Int {
        guard let node = cache[key] else {
            return -1
        }
        
        // Move to front (most recently used)
        moveToFront(node)
        return node.value
    }
    
    func put(_ key: Int, _ value: Int) {
        if let node = cache[key] {
            // Update existing
            node.value = value
            moveToFront(node)
        } else {
            // Add new node
            let newNode = Node(key, value)
            cache[key] = newNode
            addToFront(newNode)
            
            // Check capacity
            if cache.count > capacity {
                // Remove LRU (node before tail)
                if let lru = tail.prev {
                    removeNode(lru)
                    cache.removeValue(forKey: lru.key)
                }
            }
        }
    }
    
    private func addToFront(_ node: Node) {
        node.next = head.next
        node.prev = head
        head.next?.prev = node
        head.next = node
    }
    
    private func removeNode(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func moveToFront(_ node: Node) {
        removeNode(node)
        addToFront(node)
    }
}

// Usage
let lru = LRUCache(2)
lru.put(1, 1)  // cache: {1=1}
lru.put(2, 2)  // cache: {1=1, 2=2}
print(lru.get(1))    // returns 1
lru.put(3, 3)  // evicts key 2, cache: {1=1, 3=3}
print(lru.get(2))    // returns -1 (not found)
lru.put(4, 4)  // evicts key 1, cache: {3=3, 4=4}
print(lru.get(1))    // returns -1 (not found)
print(lru.get(3))    // returns 3
print(lru.get(4))    // returns 4
```

---

### Browser History Implementation

```swift
class BrowserHistory {
    class Page {
        var url: String
        var prev: Page?
        var next: Page?
        
        init(_ url: String) {
            self.url = url
        }
    }
    
    private var current: Page
    
    init(_ homepage: String) {
        current = Page(homepage)
    }
    
    func visit(_ url: String) {
        let newPage = Page(url)
        current.next = newPage
        newPage.prev = current
        current = newPage
        // Clear forward history
    }
    
    func back(_ steps: Int) -> String {
        var steps = steps
        while steps > 0 && current.prev != nil {
            current = current.prev!
            steps -= 1
        }
        return current.url
    }
    
    func forward(_ steps: Int) -> String {
        var steps = steps
        while steps > 0 && current.next != nil {
            current = current.next!
            steps -= 1
        }
        return current.url
    }
}

// Usage
let browser = BrowserHistory("leetcode.com")
browser.visit("google.com")
browser.visit("facebook.com")
browser.visit("youtube.com")
print(browser.back(1))      // "facebook.com"
print(browser.back(1))      // "google.com"
print(browser.forward(1))   // "facebook.com"
browser.visit("linkedin.com")
print(browser.forward(2))   // "linkedin.com"
print(browser.back(2))      // "google.com"
print(browser.back(7))      // "leetcode.com"
```

---

### Flattening a Multilevel Doubly Linked List

```swift
// Flatten a list with child pointers
class MultiLevelNode {
    var val: Int
    var prev: MultiLevelNode?
    var next: MultiLevelNode?
    var child: MultiLevelNode?
    
    init(_ val: Int) {
        self.val = val
    }
}

func flatten(_ head: MultiLevelNode?) -> MultiLevelNode? {
    guard let head = head else { return nil }
    
    var current: MultiLevelNode? = head
    
    while current != nil {
        if let child = current?.child {
            // Save next
            let next = current?.next
            
            // Connect to child
            current?.next = child
            child.prev = current
            current?.child = nil
            
            // Find tail of child list
            var tail = child
            while tail.next != nil {
                tail = tail.next!
            }
            
            // Connect tail to next
            tail.next = next
            next?.prev = tail
        }
        
        current = current?.next
    }
    
    return head
}
```

---

### Design Deque (Double-Ended Queue)

```swift
class MyDeque {
    private var head: DoublyListNode?
    private var tail: DoublyListNode?
    private var size: Int = 0
    
    func addFirst(_ value: Int) {
        let newNode = DoublyListNode(value)
        
        if head == nil {
            head = newNode
            tail = newNode
        } else {
            newNode.next = head
            head?.prev = newNode
            head = newNode
        }
        
        size += 1
    }
    
    func addLast(_ value: Int) {
        let newNode = DoublyListNode(value)
        
        if tail == nil {
            head = newNode
            tail = newNode
        } else {
            tail?.next = newNode
            newNode.prev = tail
            tail = newNode
        }
        
        size += 1
    }
    
    func removeFirst() -> Int? {
        guard let head = head else { return nil }
        
        let value = head.val
        self.head = head.next
        self.head?.prev = nil
        
        if self.head == nil {
            tail = nil
        }
        
        size -= 1
        return value
    }
    
    func removeLast() -> Int? {
        guard let tail = tail else { return nil }
        
        let value = tail.val
        self.tail = tail.prev
        self.tail?.next = nil
        
        if self.tail == nil {
            head = nil
        }
        
        size -= 1
        return value
    }
    
    func peekFirst() -> Int? {
        return head?.val
    }
    
    func peekLast() -> Int? {
        return tail?.val
    }
    
    var count: Int {
        return size
    }
    
    var isEmpty: Bool {
        return size == 0
    }
}
```

### Tips & Tricks

1. **Dummy nodes**: Use dummy head/tail to simplify edge cases
2. **Four pointer updates**: When inserting/deleting, update both prev and next
3. **Bidirectional traversal**: Optimize by starting from closer end
4. **Cache problems**: Doubly linked list + hash map = O(1) operations
5. **Draw diagrams**: Always sketch pointer changes before coding
6. **Null checks**: More pointers = more null checks!

<a name="chapter-10"></a>
## Chapter 10: Advanced Linked List Problems

Time to level up with challenging linked list problems!

---

### Problem 1: Reorder List

```swift
// Reorder: L0 → L1 → L2 → ... → Ln-1 → Ln
// To:      L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...
// Example: 1 -> 2 -> 3 -> 4 -> 5
// Result:  1 -> 5 -> 2 -> 4 -> 3

func reorderList(_ head: ListNode?) {
    guard head != nil && head?.next != nil else { return }
    
    // Step 1: Find middle
    var slow = head
    var fast = head
    
    while fast?.next != nil && fast?.next?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    // Step 2: Reverse second half
    var prev: ListNode? = nil
    var current = slow?.next
    slow?.next = nil  // Split list
    
    while current != nil {
        let next = current?.next
        current?.next = prev
        prev = current
        current = next
    }
    
    // Step 3: Merge two halves
    var first = head
    var second = prev
    
    while second != nil {
        let temp1 = first?.next
        let temp2 = second?.next
        
        first?.next = second
        second?.next = temp1
        
        first = temp1
        second = temp2
    }
}

// Test
let list = ListNode(1)
list.next = ListNode(2)
list.next?.next = ListNode(3)
list.next?.next?.next = ListNode(4)
list.next?.next?.next?.next = ListNode(5)
reorderList(list)
// Result: 1 -> 5 -> 2 -> 4 -> 3
```

---

### Problem 2: Add Two Numbers

```swift
// Add two numbers represented as linked lists (reversed)
// 2 -> 4 -> 3  (342)
// 5 -> 6 -> 4  (465)
// Result: 7 -> 0 -> 8  (807)

func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    var current = dummy
    var carry = 0
    var l1 = l1
    var l2 = l2
    
    while l1 != nil || l2 != nil || carry > 0 {
        let sum = (l1?.val ?? 0) + (l2?.val ?? 0) + carry
        carry = sum / 10
        
        current.next = ListNode(sum % 10)
        current = current.next!
        
        l1 = l1?.next
        l2 = l2?.next
    }
    
    return dummy.next
}

// If numbers are stored forward (not reversed)
func addTwoNumbersII(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    // Use stacks
    var stack1 = [Int]()
    var stack2 = [Int]()
    
    var current = l1
    while current != nil {
        stack1.append(current!.val)
        current = current?.next
    }
    
    current = l2
    while current != nil {
        stack2.append(current!.val)
        current = current?.next
    }
    
    var carry = 0
    var result: ListNode? = nil
    
    while !stack1.isEmpty || !stack2.isEmpty || carry > 0 {
        let sum = (stack1.popLast() ?? 0) + (stack2.popLast() ?? 0) + carry
        carry = sum / 10
        
        let newNode = ListNode(sum % 10)
        newNode.next = result
        result = newNode
    }
    
    return result
}
```

---

### Problem 3: Copy List with Random Pointer

```swift
// Deep copy a linked list where each node has a random pointer

class RandomNode {
    var val: Int
    var next: RandomNode?
    var random: RandomNode?
    
    init(_ val: Int) {
        self.val = val
    }
}

func copyRandomList(_ head: RandomNode?) -> RandomNode? {
    guard let head = head else { return nil }
    
    var map = [RandomNode: RandomNode]()
    
    // First pass: Create all nodes
    var current: RandomNode? = head
    while current != nil {
        map[current!] = RandomNode(current!.val)
        current = current?.next
    }
    
    // Second pass: Set next and random pointers
    current = head
    while current != nil {
        if let next = current?.next {
            map[current!]?.next = map[next]
        }
        if let random = current?.random {
            map[current!]?.random = map[random]
        }
        current = current?.next
    }
    
    return map[head]
}

// Alternative: O(1) space solution
func copyRandomListConstantSpace(_ head: RandomNode?) -> RandomNode? {
    guard let head = head else { return nil }
    
    // Step 1: Interweave original and copy nodes
    var current: RandomNode? = head
    while current != nil {
        let copy = RandomNode(current!.val)
        copy.next = current?.next
        current?.next = copy
        current = copy.next
    }
    
    // Step 2: Set random pointers for copies
    current = head
    while current != nil {
        if let random = current?.random {
            current?.next?.random = random.next
        }
        current = current?.next?.next
    }
    
    // Step 3: Separate lists
    let dummy = RandomNode(0)
    var copyCurrent = dummy
    current = head
    
    while current != nil {
        let copy = current?.next
        current?.next = copy?.next
        copyCurrent.next = copy
        
        current = current?.next
        copyCurrent = copyCurrent.next!
    }
    
    return dummy.next
}
```

---

### Problem 4: Merge K Sorted Lists

```swift
// Merge k sorted linked lists into one sorted list

// Approach 1: Using min heap - O(N log k)
func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
    guard !lists.isEmpty else { return nil }
    
    // Min heap simulation with array
    var heap = [(val: Int, node: ListNode)]()
    
    // Add first node of each list
    for list in lists {
        if let node = list {
            heap.append((node.val, node))
        }
    }
    
    // Sort heap
    heap.sort { $0.val < $1.val }
    
    let dummy = ListNode(0)
    var current = dummy
    
    while !heap.isEmpty {
        // Remove min
        let minPair = heap.removeFirst()
        let node = minPair.node
        
        current.next = node
        current = current.next!
        
        // Add next node from same list
        if let next = node.next {
            var inserted = false
            for i in 0..<heap.count {
                if next.val < heap[i].val {
                    heap.insert((next.val, next), at: i)
                    inserted = true
                    break
                }
            }
            if !inserted {
                heap.append((next.val, next))
            }
        }
    }
    
    return dummy.next
}

// Approach 2: Divide and conquer - O(N log k)
func mergeKListsDivideConquer(_ lists: [ListNode?]) -> ListNode? {
    guard !lists.isEmpty else { return nil }
    
    return merge(lists, 0, lists.count - 1)
    
    func merge(_ lists: [ListNode?], _ left: Int, _ right: Int) -> ListNode? {
        if left == right {
            return lists[left]
        }
        
        if left > right {
            return nil
        }
        
        let mid = left + (right - left) / 2
        let l1 = merge(lists, left, mid)
        let l2 = merge(lists, mid + 1, right)
        
        return mergeTwoLists(l1, l2)
    }
    
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let dummy = ListNode(0)
        var current = dummy
        var l1 = l1
        var l2 = l2
        
        while l1 != nil && l2 != nil {
            if l1!.val <= l2!.val {
                current.next = l1
                l1 = l1?.next
            } else {
                current.next = l2
                l2 = l2?.next
            }
            current = current.next!
        }
        
        current.next = l1 ?? l2
        return dummy.next
    }
}
```

---

### Problem 5: Rotate List

```swift
// Rotate list to the right by k places
// [1, 2, 3, 4, 5], k = 2 → [4, 5, 1, 2, 3]

func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
    guard let head = head, k > 0 else { return head }
    
    // Find length and tail
    var length = 1
    var tail = head
    while tail.next != nil {
        tail = tail.next!
        length += 1
    }
    
    // Calculate actual rotations
    let k = k % length
    if k == 0 { return head }
    
    // Find new tail (length - k - 1 from head)
    var newTail = head
    for _ in 0..<(length - k - 1) {
        newTail = newTail.next!
    }
    
    let newHead = newTail.next
    newTail.next = nil
    tail.next = head
    
    return newHead
}
```

---

### Problem 6: Partition List

```swift
// Partition list around value x
// All nodes < x come before nodes >= x
// [1, 4, 3, 2, 5, 2], x = 3 → [1, 2, 2, 4, 3, 5]

func partition(_ head: ListNode?, _ x: Int) -> ListNode? {
    let beforeDummy = ListNode(0)
    let afterDummy = ListNode(0)
    
    var before = beforeDummy
    var after = afterDummy
    var current = head
    
    while current != nil {
        if current!.val < x {
            before.next = current
            before = before.next!
        } else {
            after.next = current
            after = after.next!
        }
        current = current?.next
    }
    
    after.next = nil
    before.next = afterDummy.next
    
    return beforeDummy.next
}
```

---

### Problem 7: Reverse Nodes in K-Group

```swift
// Reverse nodes in groups of k
// [1, 2, 3, 4, 5], k = 2 → [2, 1, 4, 3, 5]
// [1, 2, 3, 4, 5], k = 3 → [3, 2, 1, 4, 5]

func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
    guard let head = head, k > 1 else { return head }
    
    // Check if we have k nodes
    var count = 0
    var current: ListNode? = head
    while current != nil && count < k {
        current = current?.next
        count += 1
    }
    
    if count < k {
        return head  // Not enough nodes to reverse
    }
    
    // Reverse first k nodes
    var prev: ListNode? = nil
    current = head
    
    for _ in 0..<k {
        let next = current?.next
        current?.next = prev
        prev = current
        current = next
    }
    
    // Recursively reverse remaining groups
    head.next = reverseKGroup(current, k)
    
    return prev
}
```

---

### Problem 8: Sort List

```swift
// Sort linked list in O(n log n) time, O(1) space
// Use merge sort

func sortList(_ head: ListNode?) -> ListNode? {
    guard let head = head, head.next != nil else {
        return head
    }
    
    // Find middle
    var slow: ListNode? = head
    var fast: ListNode? = head
    var prev: ListNode? = nil
    
    while fast != nil && fast?.next != nil {
        prev = slow
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    // Split list
    prev?.next = nil
    
    // Recursively sort both halves
    let left = sortList(head)
    let right = sortList(slow)
    
    // Merge sorted halves
    return merge(left, right)
    
    func merge(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let dummy = ListNode(0)
        var current = dummy
        var l1 = l1
        var l2 = l2
        
        while l1 != nil && l2 != nil {
            if l1!.val <= l2!.val {
                current.next = l1
                l1 = l1?.next
            } else {
                current.next = l2
                l2 = l2?.next
            }
            current = current.next!
        }
        
        current.next = l1 ?? l2
        return dummy.next
    }
}
```

---

### Problem 9: Swap Nodes in Pairs

```swift
// Swap adjacent nodes
// [1, 2, 3, 4] → [2, 1, 4, 3]

func swapPairs(_ head: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    dummy.next = head
    var prev = dummy
    
    while prev.next != nil && prev.next?.next != nil {
        let first = prev.next
        let second = prev.next?.next
        
        // Swap
        prev.next = second
        first?.next = second?.next
        second?.next = first
        
        prev = first!
    }
    
    return dummy.next
}

// Recursive version
func swapPairsRecursive(_ head: ListNode?) -> ListNode? {
    guard let first = head, let second = head.next else {
        return head
    }
    
    first.next = swapPairsRecursive(second.next)
    second.next = first
    
    return second
}
```

---

### Problem 10: Odd Even Linked List

```swift
// Group odd indices together, then even indices
// [1, 2, 3, 4, 5] → [1, 3, 5, 2, 4]

func oddEvenList(_ head: ListNode?) -> ListNode? {
    guard let head = head else { return nil }
    
    var odd = head
    var even = head.next
    let evenHead = even
    
    while even != nil && even?.next != nil {
        odd.next = even?.next
        odd = odd.next!
        
        even?.next = odd.next
        even = even?.next
    }
    
    odd.next = evenHead
    return head
}
```

---

### Advanced Techniques Summary

1. **Dummy node**: Simplifies edge cases, especially for head operations
2. **Two/Three pointers**: Fast & slow, prev & current, etc.
3. **Recursion**: Elegant for reversals and tree-like problems
4. **Hash maps**: For random pointers, cycle detection start
5. **Stacks**: For reversed number addition, palindrome check
6. **Merge sort**: O(n log n) sorting for linked lists
7. **Divide & conquer**: Merge k lists, sort list
8. **Runner technique**: K-group reversal, rotate list

### Debugging Tips

1. **Draw it**: Visualize pointer movements
2. **Edge cases**: Test with 0, 1, 2 nodes
3. **Dummy nodes**: Prevent null pointer errors
4. **Preserve connections**: Don't lose references before reassigning
5. **Test modifications**: Ensure both directions updated (doubly linked)

### Common Mistakes to Avoid

❌ Losing reference to head  
❌ Forgetting to update both pointers in doubly linked  
❌ Off-by-one errors in k-group problems  
❌ Not checking for null before dereferencing  
❌ Forgetting to break cycles when reversing  
❌ Not using dummy nodes when head can change  

---

**🎯 Practice Problems:**
1. Delete Node in a Linked List (O(1))
2. Linked List Components
3. Split Linked List in Parts
4. Next Greater Node in Linked List
5. Remove Zero Sum Consecutive Nodes

<a name="chapter-11"></a>
## Chapter 11: Stack Implementation & Problems

Stacks follow the **Last-In-First-Out (LIFO)** principle. Think of a stack of plates - you add and remove from the top!

### Stack Basics

```swift
// Stack operations
// push(item) - Add to top - O(1)
// pop() - Remove from top - O(1)
// peek()/top() - View top without removing - O(1)
// isEmpty() - Check if empty - O(1)
// size() - Get count - O(1)
```

### Stack Implementation Using Array

```swift
struct Stack<T> {
    private var elements: [T] = []
    
    var isEmpty: Bool {
        return elements.isEmpty
    }
    
    var count: Int {
        return elements.count
    }
    
    var peek: T? {
        return elements.last
    }
    
    mutating func push(_ element: T) {
        elements.append(element)
    }
    
    @discardableResult
    mutating func pop() -> T? {
        return elements.popLast()
    }
    
    func printStack() {
        print("Top -> ", terminator: "")
        for i in stride(from: elements.count - 1, through: 0, by: -1) {
            print(elements[i], terminator: " ")
        }
        print()
    }
}

// Usage
var stack = Stack<Int>()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.peek ?? "Empty")  // 3
stack.pop()
print(stack.peek ?? "Empty")  // 2
```

### Stack Implementation Using Linked List

```swift
class StackNode<T> {
    var value: T
    var next: StackNode?
    
    init(_ value: T) {
        self.value = value
    }
}

class LinkedStack<T> {
    private var top: StackNode<T>?
    private(set) var count = 0
    
    var isEmpty: Bool {
        return top == nil
    }
    
    var peek: T? {
        return top?.value
    }
    
    func push(_ value: T) {
        let newNode = StackNode(value)
        newNode.next = top
        top = newNode
        count += 1
    }
    
    @discardableResult
    func pop() -> T? {
        let value = top?.value
        top = top?.next
        if value != nil {
            count -= 1
        }
        return value
    }
}
```

---

### Classic Stack Problems

#### Problem 1: Valid Parentheses

```swift
// Check if parentheses are valid
// "()" → true
// "()[]{}" → true
// "(]" → false
// "([)]" → false

func isValid(_ s: String) -> Bool {
    var stack = [Character]()
    let pairs: [Character: Character] = [")": "(", "}": "{", "]": "["]
    
    for char in s {
        if pairs.values.contains(char) {
            // Opening bracket
            stack.append(char)
        } else if let opening = pairs[char] {
            // Closing bracket
            if stack.isEmpty || stack.last != opening {
                return false
            }
            stack.removeLast()
        }
    }
    
    return stack.isEmpty
}

print(isValid("()"))        // true
print(isValid("()[]{}"))    // true
print(isValid("(]"))        // false
print(isValid("([)]"))      // false
print(isValid("{[]}"))      // true
```

#### Problem 2: Min Stack

```swift
// Design stack that supports push, pop, top, and getMin in O(1)

class MinStack {
    private var stack: [(val: Int, min: Int)] = []
    
    func push(_ val: Int) {
        let currentMin = stack.isEmpty ? val : min(val, stack.last!.min)
        stack.append((val, currentMin))
    }
    
    func pop() {
        stack.removeLast()
    }
    
    func top() -> Int {
        return stack.last!.val
    }
    
    func getMin() -> Int {
        return stack.last!.min
    }
}

// Alternative: Using two stacks
class MinStack2 {
    private var stack: [Int] = []
    private var minStack: [Int] = []
    
    func push(_ val: Int) {
        stack.append(val)
        if minStack.isEmpty || val <= minStack.last! {
            minStack.append(val)
        }
    }
    
    func pop() {
        let val = stack.removeLast()
        if val == minStack.last! {
            minStack.removeLast()
        }
    }
    
    func top() -> Int {
        return stack.last!
    }
    
    func getMin() -> Int {
        return minStack.last!
    }
}

// Usage
let minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())  // -3
minStack.pop()
print(minStack.top())     // 0
print(minStack.getMin())  // -2
```

#### Problem 3: Evaluate Reverse Polish Notation

```swift
// Evaluate expression in RPN
// ["2", "1", "+", "3", "*"] → ((2 + 1) * 3) = 9
// ["4", "13", "5", "/", "+"] → (4 + (13 / 5)) = 6

func evalRPN(_ tokens: [String]) -> Int {
    var stack = [Int]()
    
    for token in tokens {
        if let num = Int(token) {
            stack.append(num)
        } else {
            let b = stack.removeLast()
            let a = stack.removeLast()
            
            switch token {
            case "+":
                stack.append(a + b)
            case "-":
                stack.append(a - b)
            case "*":
                stack.append(a * b)
            case "/":
                stack.append(a / b)
            default:
                break
            }
        }
    }
    
    return stack.last!
}

print(evalRPN(["2", "1", "+", "3", "*"]))  // 9
print(evalRPN(["4", "13", "5", "/", "+"]))  // 6
```

#### Problem 4: Daily Temperatures

```swift
// Find number of days until warmer temperature
// [73, 74, 75, 71, 69, 72, 76, 73]
// → [1, 1, 4, 2, 1, 1, 0, 0]

func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
    var result = Array(repeating: 0, count: temperatures.count)
    var stack = [Int]()  // Store indices
    
    for i in 0..<temperatures.count {
        while !stack.isEmpty && temperatures[i] > temperatures[stack.last!] {
            let prevIndex = stack.removeLast()
            result[prevIndex] = i - prevIndex
        }
        stack.append(i)
    }
    
    return result
}

print(dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
// [1, 1, 4, 2, 1, 1, 0, 0]
```

#### Problem 5: Next Greater Element

```swift
// Find next greater element for each element
// [4, 5, 2, 25] → [5, 25, 25, -1]

func nextGreaterElement(_ nums: [Int]) -> [Int] {
    var result = Array(repeating: -1, count: nums.count)
    var stack = [Int]()  // Store indices
    
    for i in 0..<nums.count {
        while !stack.isEmpty && nums[i] > nums[stack.last!] {
            let prevIndex = stack.removeLast()
            result[prevIndex] = nums[i]
        }
        stack.append(i)
    }
    
    return result
}

print(nextGreaterElement([4, 5, 2, 25]))  // [5, 25, 25, -1]

// For circular array
func nextGreaterElementsCircular(_ nums: [Int]) -> [Int] {
    let n = nums.count
    var result = Array(repeating: -1, count: n)
    var stack = [Int]()
    
    // Loop twice to handle circular
    for i in 0..<(2 * n) {
        let index = i % n
        
        while !stack.isEmpty && nums[index] > nums[stack.last!] {
            result[stack.removeLast()] = nums[index]
        }
        
        if i < n {
            stack.append(index)
        }
    }
    
    return result
}

print(nextGreaterElementsCircular([1, 2, 1]))  // [2, -1, 2]
```

#### Problem 6: Simplify Path

```swift
// Simplify Unix-style file path
// "/home/" → "/home"
// "/../" → "/"
// "/home//foo/" → "/home/foo"
// "/a/./b/../../c/" → "/c"

func simplifyPath(_ path: String) -> String {
    var stack = [String]()
    let components = path.split(separator: "/").map(String.init)
    
    for component in components {
        if component == ".." {
            stack.popLast()
        } else if component != "." && !component.isEmpty {
            stack.append(component)
        }
    }
    
    return "/" + stack.joined(separator: "/")
}

print(simplifyPath("/home/"))           // "/home"
print(simplifyPath("/../"))             // "/"
print(simplifyPath("/home//foo/"))      // "/home/foo"
print(simplifyPath("/a/./b/../../c/"))  // "/c"
```

#### Problem 7: Decode String

```swift
// Decode encoded string
// "3[a]2[bc]" → "aaabcbc"
// "3[a2[c]]" → "accaccacc"
// "2[abc]3[cd]ef" → "abcabccdcdcdef"

func decodeString(_ s: String) -> String {
    var countStack = [Int]()
    var stringStack = [String]()
    var currentString = ""
    var currentNum = 0
    
    for char in s {
        if char.isNumber {
            currentNum = currentNum * 10 + Int(String(char))!
        } else if char == "[" {
            countStack.append(currentNum)
            stringStack.append(currentString)
            currentNum = 0
            currentString = ""
        } else if char == "]" {
            let prevString = stringStack.removeLast()
            let count = countStack.removeLast()
            currentString = prevString + String(repeating: currentString, count: count)
        } else {
            currentString.append(char)
        }
    }
    
    return currentString
}

print(decodeString("3[a]2[bc]"))      // "aaabcbc"
print(decodeString("3[a2[c]]"))       // "accaccacc"
print(decodeString("2[abc]3[cd]ef"))  // "abcabccdcdcdef"
```

#### Problem 8: Remove K Digits

```swift
// Remove k digits to make smallest number
// "1432219", k = 3 → "1219"
// "10200", k = 1 → "200"

func removeKdigits(_ num: String, _ k: Int) -> String {
    var k = k
    var stack = [Character]()
    
    for digit in num {
        // Remove larger digits from stack
        while k > 0 && !stack.isEmpty && stack.last! > digit {
            stack.removeLast()
            k -= 1
        }
        stack.append(digit)
    }
    
    // Remove remaining k digits from end
    while k > 0 {
        stack.removeLast()
        k -= 1
    }
    
    // Remove leading zeros
    while !stack.isEmpty && stack.first == "0" {
        stack.removeFirst()
    }
    
    return stack.isEmpty ? "0" : String(stack)
}

print(removeKdigits("1432219", 3))  // "1219"
print(removeKdigits("10200", 1))    // "200"
print(removeKdigits("10", 2))       // "0"
```

#### Problem 9: Asteroid Collision

```swift
// Asteroids moving left (-) and right (+) collide
// [5, 10, -5] → [5, 10] (10 destroys -5)
// [8, -8] → [] (both destroyed)
// [10, 2, -5] → [10] (10 destroys -5, 2 destroyed by 10)

func asteroidCollision(_ asteroids: [Int]) -> [Int] {
    var stack = [Int]()
    
    for asteroid in asteroids {
        var alive = true
        
        while alive && asteroid < 0 && !stack.isEmpty && stack.last! > 0 {
            let top = stack.last!
            
            if abs(asteroid) > top {
                stack.removeLast()  // Top destroyed
            } else if abs(asteroid) == top {
                stack.removeLast()  // Both destroyed
                alive = false
            } else {
                alive = false  // Current destroyed
            }
        }
        
        if alive {
            stack.append(asteroid)
        }
    }
    
    return stack
}

print(asteroidCollision([5, 10, -5]))    // [5, 10]
print(asteroidCollision([8, -8]))        // []
print(asteroidCollision([10, 2, -5]))    // [10]
print(asteroidCollision([-2, -1, 1, 2])) // [-2, -1, 1, 2]
```

#### Problem 10: Basic Calculator

```swift
// Evaluate expression with +, -, (, )
// "1 + 1" → 2
// " 2-1 + 2 " → 3
// "(1+(4+5+2)-3)+(6+8)" → 23

func calculate(_ s: String) -> Int {
    var stack = [Int]()
    var num = 0
    var sign = 1
    var result = 0
    
    for char in s {
        if char.isNumber {
            num = num * 10 + Int(String(char))!
        } else if char == "+" {
            result += sign * num
            num = 0
            sign = 1
        } else if char == "-" {
            result += sign * num
            num = 0
            sign = -1
        } else if char == "(" {
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        } else if char == ")" {
            result += sign * num
            num = 0
            result *= stack.removeLast()  // sign
            result += stack.removeLast()  // previous result
        }
    }
    
    result += sign * num
    return result
}

print(calculate("1 + 1"))                  // 2
print(calculate(" 2-1 + 2 "))             // 3
print(calculate("(1+(4+5+2)-3)+(6+8)"))   // 23
```

---

### Stack Patterns & When to Use

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Matching/Balancing** | Parentheses, brackets | Valid parentheses |
| **Monotonic Stack** | Next greater/smaller | Daily temperatures |
| **Expression Evaluation** | Math expressions | Calculator, RPN |
| **String Manipulation** | Decode, simplify | Decode string, simplify path |
| **Collision/Interaction** | Elements colliding | Asteroid collision |
| **Backtracking** | Undo operations | Browser history |

### Stack Tricks & Tips

1. **Use indices, not values**: Store indices in stack for position tracking
2. **Monotonic stack**: Maintain increasing/decreasing order
3. **Dummy values**: Sometimes helps with edge cases
4. **Two stacks**: One for values, one for min/max/operators
5. **Stack + hash map**: For more complex state tracking
6. **Process on pop**: Often the answer is revealed when popping
7. **Clear condition**: Know exactly when to push vs pop

---

<a name="chapter-12"></a>
## Chapter 12: Queue Implementation & Problems

Queues follow the **First-In-First-Out (FIFO)** principle. Think of a line at a store!

### Queue Basics

```swift
// Queue operations
// enqueue(item) - Add to back - O(1)
// dequeue() - Remove from front - O(1)
// peek()/front() - View front without removing - O(1)
// isEmpty() - Check if empty - O(1)
// size() - Get count - O(1)
```

### Queue Implementation Using Array

```swift
struct Queue<T> {
    private var elements: [T] = []
    
    var isEmpty: Bool {
        return elements.isEmpty
    }
    
    var count: Int {
        return elements.count
    }
    
    var peek: T? {
        return elements.first
    }
    
    mutating func enqueue(_ element: T) {
        elements.append(element)
    }
    
    @discardableResult
    mutating func dequeue() -> T? {
        guard !isEmpty else { return nil }
        return elements.removeFirst()
    }
}

// Usage
var queue = Queue<Int>()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.peek ?? "Empty")  // 1
queue.dequeue()
print(queue.peek ?? "Empty")  // 2
```

⚠️ **Problem**: `removeFirst()` is O(n) in Swift arrays!

### Efficient Queue (Circular Buffer)

```swift
struct EfficientQueue<T> {
    private var array: [T?]
    private var head = 0
    private var tail = 0
    private var count = 0
    
    init(capacity: Int) {
        array = Array(repeating: nil, count: capacity)
    }
    
    var isEmpty: Bool {
        return count == 0
    }
    
    var isFull: Bool {
        return count == array.count
    }
    
    var peek: T? {
        return isEmpty ? nil : array[head]
    }
    
    mutating func enqueue(_ element: T) -> Bool {
        guard !isFull else { return false }
        
        array[tail] = element
        tail = (tail + 1) % array.count
        count += 1
        return true
    }
    
    @discardableResult
    mutating func dequeue() -> T? {
        guard !isEmpty else { return nil }
        
        let element = array[head]
        array[head] = nil
        head = (head + 1) % array.count
        count -= 1
        return element
    }
}
```

### Queue Using Two Stacks

```swift
// Amortized O(1) operations
struct QueueWithStacks<T> {
    private var inStack: [T] = []
    private var outStack: [T] = []
    
    var isEmpty: Bool {
        return inStack.isEmpty && outStack.isEmpty
    }
    
    var count: Int {
        return inStack.count + outStack.count
    }
    
    var peek: T? {
        if outStack.isEmpty {
            outStack = inStack.reversed()
            inStack.removeAll()
        }
        return outStack.last
    }
    
    mutating func enqueue(_ element: T) {
        inStack.append(element)
    }
    
    @discardableResult
    mutating func dequeue() -> T? {
        if outStack.isEmpty {
            outStack = inStack.reversed()
            inStack.removeAll()
        }
        return outStack.popLast()
    }
}
```

### Queue Using Linked List

```swift
class QueueNode<T> {
    var value: T
    var next: QueueNode?
    
    init(_ value: T) {
        self.value = value
    }
}

class LinkedQueue<T> {
    private var head: QueueNode<T>?
    private var tail: QueueNode<T>?
    private(set) var count = 0
    
    var isEmpty: Bool {
        return head == nil
    }
    
    var peek: T? {
        return head?.value
    }
    
    func enqueue(_ value: T) {
        let newNode = QueueNode(value)
        
        if let tail = tail {
            tail.next = newNode
        } else {
            head = newNode
        }
        
        tail = newNode
        count += 1
    }
    
    @discardableResult
    func dequeue() -> T? {
        guard let head = head else { return nil }
        
        self.head = head.next
        if self.head == nil {
            tail = nil
        }
        
        count -= 1
        return head.value
    }
}
```

---

### Classic Queue Problems

#### Problem 1: Implement Stack Using Queues

```swift
// Implement stack using only queues

class MyStack {
    private var queue = [Int]()
    
    func push(_ x: Int) {
        queue.append(x)
        
        // Rotate queue to make new element front
        for _ in 0..<(queue.count - 1) {
            queue.append(queue.removeFirst())
        }
    }
    
    func pop() -> Int {
        return queue.removeFirst()
    }
    
    func top() -> Int {
        return queue.first!
    }
    
    func empty() -> Bool {
        return queue.isEmpty
    }
}
```

#### Problem 2: Number of Recent Calls

```swift
// Count requests in last 3000ms

class RecentCounter {
    private var requests: [Int] = []
    
    func ping(_ t: Int) -> Int {
        requests.append(t)
        
        // Remove requests older than 3000ms
        while !requests.isEmpty && requests.first! < t - 3000 {
            requests.removeFirst()
        }
        
        return requests.count
    }
}

let counter = RecentCounter()
print(counter.ping(1))     // 1
print(counter.ping(100))   // 2
print(counter.ping(3001))  // 3
print(counter.ping(3002))  // 3
```

#### Problem 3: Moving Average from Data Stream

```swift
class MovingAverage {
    private var queue: [Int] = []
    private var size: Int
    private var sum: Double = 0
    
    init(_ size: Int) {
        self.size = size
    }
    
    func next(_ val: Int) -> Double {
        queue.append(val)
        sum += Double(val)
        
        if queue.count > size {
            sum -= Double(queue.removeFirst())
        }
        
        return sum / Double(queue.count)
    }
}

let ma = MovingAverage(3)
print(ma.next(1))   // 1.0
print(ma.next(10))  // 5.5
print(ma.next(3))   // 4.66667
print(ma.next(5))   // 6.0
```

#### Problem 4: Perfect Squares (BFS)

```swift
// Find minimum number of perfect squares that sum to n
// n = 12 → 3 (4 + 4 + 4)
// n = 13 → 2 (4 + 9)

func numSquares(_ n: Int) -> Int {
    var queue = [(num: n, steps: 0)]
    var visited = Set<Int>()
    visited.insert(n)
    
    while !queue.isEmpty {
        let (num, steps) = queue.removeFirst()
        
        if num == 0 {
            return steps
        }
        
        var i = 1
        while i * i <= num {
            let next = num - i * i
            
            if !visited.contains(next) {
                visited.insert(next)
                queue.append((next, steps + 1))
            }
            
            i += 1
        }
    }
    
    return 0
}

print(numSquares(12))  // 3
print(numSquares(13))  // 2
```

#### Problem 5: Design Circular Queue

```swift
class MyCircularQueue {
    private var array: [Int?]
    private var head = 0
    private var tail = 0
    private var count = 0
    private let capacity: Int
    
    init(_ k: Int) {
        capacity = k
        array = Array(repeating: nil, count: k)
    }
    
    func enQueue(_ value: Int) -> Bool {
        guard !isFull() else { return false }
        
        array[tail] = value
        tail = (tail + 1) % capacity
        count += 1
        return true
    }
    
    func deQueue() -> Bool {
        guard !isEmpty() else { return false }
        
        array[head] = nil
        head = (head + 1) % capacity
        count -= 1
        return true
    }
    
    func Front() -> Int {
        return isEmpty() ? -1 : array[head]!
    }
    
    func Rear() -> Int {
        if isEmpty() { return -1 }
        let rearIndex = (tail - 1 + capacity) % capacity
        return array[rearIndex]!
    }
    
    func isEmpty() -> Bool {
        return count == 0
    }
    
    func isFull() -> Bool {
        return count == capacity
    }
}
```

#### Problem 6: Task Scheduler

```swift
// Schedule tasks with cooldown period
// tasks = ["A","A","A","B","B","B"], n = 2 → 8
// Output: A -> B -> idle -> A -> B -> idle -> A -> B

func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
    var freq = [Character: Int]()
    
    for task in tasks {
        freq[task, default: 0] += 1
    }
    
    let maxFreq = freq.values.max()!
    let maxCount = freq.values.filter { $0 == maxFreq }.count
    
    let intervals = (maxFreq - 1) * (n + 1) + maxCount
    
    return max(intervals, tasks.count)
}

print(leastInterval(["A","A","A","B","B","B"], 2))  // 8
print(leastInterval(["A","A","A","A","A","A","B","C","D","E","F","G"], 2))  // 16
```

#### Problem 7: First Unique Character in Stream

```swift
class FirstUnique {
    private var queue: [Int] = []
    private var freq: [Int: Int] = [:]
    
    init(_ nums: [Int]) {
        for num in nums {
            add(num)
        }
    }
    
    func showFirstUnique() -> Int {
        // Remove non-unique from front
        while !queue.isEmpty && freq[queue.first!]! > 1 {
            queue.removeFirst()
        }
        
        return queue.isEmpty ? -1 : queue.first!
    }
    
    func add(_ value: Int) {
        freq[value, default: 0] += 1
        
        if freq[value] == 1 {
            queue.append(value)
        }
    }
}

let fu = FirstUnique([2, 3, 5])
print(fu.showFirstUnique())  // 2
fu.add(5)
print(fu.showFirstUnique())  // 2
fu.add(2)
print(fu.showFirstUnique())  // 3
fu.add(3)
print(fu.showFirstUnique())  // -1
```

#### Problem 8: Generate Binary Numbers

```swift
// Generate binary numbers from 1 to n
// n = 5 → ["1", "10", "11", "100", "101"]

func generateBinaryNumbers(_ n: Int) -> [String] {
    var result = [String]()
    var queue = ["1"]
    
    for _ in 1...n {
        let binary = queue.removeFirst()
        result.append(binary)
        
        queue.append(binary + "0")
        queue.append(binary + "1")
    }
    
    return result
}

print(generateBinaryNumbers(5))
// ["1", "10", "11", "100", "101"]
```

#### Problem 9: Sliding Window Maximum (Using Deque)

```swift
// Find maximum in each sliding window
// [1, 3, -1, -3, 5, 3, 6, 7], k = 3 → [3, 3, 5, 5, 6, 7]

func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    var result = [Int]()
    var deque = [Int]()  // Store indices
    
    for i in 0..<nums.count {
        // Remove indices outside window
        if !deque.isEmpty && deque.first! <= i - k {
            deque.removeFirst()
        }
        
        // Remove smaller elements (they'll never be max)
        while !deque.isEmpty && nums[deque.last!] < nums[i] {
            deque.removeLast()
        }
        
        deque.append(i)
        
        // Add max to result
        if i >= k - 1 {
            result.append(nums[deque.first!])
        }
    }
    
    return result
}

print(maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
// [3, 3, 5, 5, 6, 7]
```

#### Problem 10: Design Hit Counter

```swift
// Count hits in last 5 minutes (300 seconds)

class HitCounter {
    private var hits: [Int] = []
    
    func hit(_ timestamp: Int) {
        hits.append(timestamp)
    }
    
    func getHits(_ timestamp: Int) -> Int {
        // Remove hits older than 300 seconds
        while !hits.isEmpty && hits.first! <= timestamp - 300 {
            hits.removeFirst()
        }
        
        return hits.count
    }
}

let counter2 = HitCounter()
counter2.hit(1)
counter2.hit(2)
counter2.hit(3)
print(counter2.getHits(4))    // 3
counter2.hit(300)
print(counter2.getHits(300))  // 4
print(counter2.getHits(301))  // 3
```

---

### Queue Patterns & When to Use

| Pattern | Use Case | Example |
|---------|----------|---------|
| **BFS Traversal** | Level-order, shortest path | Perfect squares, tree traversal |
| **Sliding Window** | Moving window max/min | Sliding window maximum |
| **Stream Processing** | Recent events, moving avg | Hit counter, moving average |
| **Scheduling** | Task ordering, cooldown | Task scheduler |
| **State Tracking** | Unique elements in order | First unique character |
| **Generation** | Sequential generation | Binary numbers |

### Deque (Double-Ended Queue)

```swift
// Deque supports operations at both ends
struct Deque<T> {
    private var array: [T] = []
    
    var isEmpty: Bool {
        return array.isEmpty
    }
    
    var count: Int {
        return array.count
    }
    
    // Front operations
    func peekFirst() -> T? {
        return array.first
    }
    
    mutating func addFirst(_ element: T) {
        array.insert(element, at: 0)
    }
    
    mutating func removeFirst() -> T? {
        guard !isEmpty else { return nil }
        return array.removeFirst()
    }
    
    // Back operations
    func peekLast() -> T? {
        return array.last
    }
    
    mutating func addLast(_ element: T) {
        array.append(element)
    }
    
    mutating func removeLast() -> T? {
        guard !isEmpty else { return nil }
        return array.removeLast()
    }
}
```

### Queue vs Stack Comparison

| Feature | Stack | Queue |
|---------|-------|-------|
| Order | LIFO | FIFO |
| Add | push (top) | enqueue (back) |
| Remove | pop (top) | dequeue (front) |
| Use Case | Undo, recursion | Scheduling, BFS |
| Real Life | Plates, browser back | Line at store |

### Tips & Tricks

1. **BFS = Queue**: Always use queue for breadth-first search
2. **Circular buffer**: Most efficient for fixed-size queues
3. **Two stacks**: Can implement queue with amortized O(1)
4. **Deque for sliding window**: Maintain monotonic property
5. **Track timestamps**: For time-based problems
6. **Level tracking**: Use queue size to track levels in BFS
7. **Visited set**: Prevent revisiting in BFS

---

**🎯 Practice Problems:**

**Stack:**
1. Valid Parenthesis String
2. Score of Parentheses
3. Remove Duplicate Letters
4. Largest Rectangle in Histogram

**Queue:**
1. Dota2 Senate
2. Shortest Subarray with Sum at Least K
3. Jump Game VI
4. Constrained Subsequence Sum

<a name="chapter-13"></a>
## Chapter 13: Monotonic Stack/Queue

Monotonic stacks and queues are powerful techniques that maintain elements in increasing or decreasing order. They're essential for "next greater/smaller" problems!

### What is a Monotonic Stack?

A stack where elements are always in **increasing** or **decreasing** order.

```swift
// Monotonic Increasing Stack: [1, 3, 5, 7]
// When pushing 4: pop 5 and 7, then push 4 → [1, 3, 4]

// Monotonic Decreasing Stack: [9, 6, 4, 2]
// When pushing 5: pop 4 and 2, then push 5 → [9, 6, 5]
```

### When to Use Monotonic Stack?

✅ **Use when you need:**
- Next greater element
- Next smaller element
- Previous greater element
- Previous smaller element
- Maximum rectangle/histogram problems
- Stock span problems

### Monotonic Stack Template

```swift
// Template for Next Greater Element
func nextGreater(_ arr: [Int]) -> [Int] {
    var result = Array(repeating: -1, count: arr.count)
    var stack = [Int]()  // Store indices
    
    for i in 0..<arr.count {
        // Pop smaller elements (maintaining decreasing stack)
        while !stack.isEmpty && arr[i] > arr[stack.last!] {
            let prevIndex = stack.removeLast()
            result[prevIndex] = arr[i]
        }
        stack.append(i)
    }
    
    return result
}

// Template for Next Smaller Element
func nextSmaller(_ arr: [Int]) -> [Int] {
    var result = Array(repeating: -1, count: arr.count)
    var stack = [Int]()
    
    for i in 0..<arr.count {
        // Pop greater elements (maintaining increasing stack)
        while !stack.isEmpty && arr[i] < arr[stack.last!] {
            let prevIndex = stack.removeLast()
            result[prevIndex] = arr[i]
        }
        stack.append(i)
    }
    
    return result
}
```

---

### Classic Monotonic Stack Problems

#### Problem 1: Next Greater Element I

```swift
// Find next greater element in nums2 for each element in nums1
// nums1 = [4, 1, 2], nums2 = [1, 3, 4, 2]
// Output: [-1, 3, -1] (no greater for 4, 3 is greater than 1, no greater for 2)

func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
    var map = [Int: Int]()
    var stack = [Int]()
    
    // Find next greater for all elements in nums2
    for num in nums2 {
        while !stack.isEmpty && num > stack.last! {
            map[stack.removeLast()] = num
        }
        stack.append(num)
    }
    
    // Build result for nums1
    return nums1.map { map[$0, default: -1] }
}

print(nextGreaterElement([4, 1, 2], [1, 3, 4, 2]))
// [-1, 3, -1]
```

#### Problem 2: Next Greater Element II (Circular)

```swift
// Array is circular - element can wrap around
// [1, 2, 1] → [2, -1, 2]

func nextGreaterElements(_ nums: [Int]) -> [Int] {
    let n = nums.count
    var result = Array(repeating: -1, count: n)
    var stack = [Int]()
    
    // Loop twice to handle circular
    for i in 0..<(2 * n) {
        let index = i % n
        
        while !stack.isEmpty && nums[index] > nums[stack.last!] {
            result[stack.removeLast()] = nums[index]
        }
        
        // Only add to stack in first pass
        if i < n {
            stack.append(index)
        }
    }
    
    return result
}

print(nextGreaterElements([1, 2, 1]))  // [2, -1, 2]
print(nextGreaterElements([1, 2, 3, 4, 3]))  // [2, 3, 4, -1, 4]
```

#### Problem 3: Daily Temperatures (Revisited)

```swift
// How many days until warmer temperature
// [73, 74, 75, 71, 69, 72, 76, 73]
// → [1, 1, 4, 2, 1, 1, 0, 0]

func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
    var result = Array(repeating: 0, count: temperatures.count)
    var stack = [Int]()  // Monotonic decreasing
    
    for i in 0..<temperatures.count {
        while !stack.isEmpty && temperatures[i] > temperatures[stack.last!] {
            let prevIndex = stack.removeLast()
            result[prevIndex] = i - prevIndex
        }
        stack.append(i)
    }
    
    return result
}
```

#### Problem 4: Largest Rectangle in Histogram

```swift
// Find largest rectangle in histogram
// [2, 1, 5, 6, 2, 3] → 10

func largestRectangleArea(_ heights: [Int]) -> Int {
    var stack = [Int]()
    var maxArea = 0
    var heights = heights + [0]  // Add sentinel
    
    for i in 0..<heights.count {
        while !stack.isEmpty && heights[i] < heights[stack.last!] {
            let h = heights[stack.removeLast()]
            let w = stack.isEmpty ? i : i - stack.last! - 1
            maxArea = max(maxArea, h * w)
        }
        stack.append(i)
    }
    
    return maxArea
}

print(largestRectangleArea([2, 1, 5, 6, 2, 3]))  // 10
print(largestRectangleArea([2, 4]))  // 4
```

#### Problem 5: Maximal Rectangle

```swift
// Find largest rectangle of 1s in binary matrix
// [
//   ["1","0","1","0","0"],
//   ["1","0","1","1","1"],
//   ["1","1","1","1","1"],
//   ["1","0","0","1","0"]
// ] → 6

func maximalRectangle(_ matrix: [[Character]]) -> Int {
    guard !matrix.isEmpty else { return 0 }
    
    let rows = matrix.count
    let cols = matrix[0].count
    var heights = Array(repeating: 0, count: cols)
    var maxArea = 0
    
    for row in 0..<rows {
        // Update heights
        for col in 0..<cols {
            if matrix[row][col] == "1" {
                heights[col] += 1
            } else {
                heights[col] = 0
            }
        }
        
        // Calculate max rectangle for this row
        maxArea = max(maxArea, largestRectangleArea(heights))
    }
    
    return maxArea
    
    func largestRectangleArea(_ heights: [Int]) -> Int {
        var stack = [Int]()
        var maxArea = 0
        var heights = heights + [0]
        
        for i in 0..<heights.count {
            while !stack.isEmpty && heights[i] < heights[stack.last!] {
                let h = heights[stack.removeLast()]
                let w = stack.isEmpty ? i : i - stack.last! - 1
                maxArea = max(maxArea, h * w)
            }
            stack.append(i)
        }
        
        return maxArea
    }
}
```

#### Problem 6: Trapping Rain Water (Stack Approach)

```swift
// Calculate trapped water using monotonic stack
// [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1] → 6

func trapStack(_ height: [Int]) -> Int {
    var stack = [Int]()
    var water = 0
    
    for i in 0..<height.count {
        while !stack.isEmpty && height[i] > height[stack.last!] {
            let bottom = stack.removeLast()
            
            if stack.isEmpty {
                break
            }
            
            let distance = i - stack.last! - 1
            let boundedHeight = min(height[i], height[stack.last!]) - height[bottom]
            water += distance * boundedHeight
        }
        
        stack.append(i)
    }
    
    return water
}

print(trapStack([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  // 6
```

#### Problem 7: Sum of Subarray Minimums

```swift
// Sum of minimum of all subarrays
// [3, 1, 2, 4] → 17
// Subarrays: [3]=3, [1]=1, [2]=2, [4]=4, [3,1]=1, [1,2]=1, [2,4]=2, [3,1,2]=1, [1,2,4]=1, [3,1,2,4]=1
// Sum = 3+1+2+4+1+1+2+1+1+1 = 17

func sumSubarrayMins(_ arr: [Int]) -> Int {
    let mod = 1_000_000_007
    let n = arr.count
    
    // Find previous less element
    var prevLess = Array(repeating: -1, count: n)
    var stack = [Int]()
    
    for i in 0..<n {
        while !stack.isEmpty && arr[stack.last!] > arr[i] {
            stack.removeLast()
        }
        prevLess[i] = stack.isEmpty ? -1 : stack.last!
        stack.append(i)
    }
    
    // Find next less element
    var nextLess = Array(repeating: n, count: n)
    stack.removeAll()
    
    for i in 0..<n {
        while !stack.isEmpty && arr[stack.last!] > arr[i] {
            nextLess[stack.removeLast()] = i
        }
        stack.append(i)
    }
    
    // Calculate sum
    var result = 0
    for i in 0..<n {
        let left = i - prevLess[i]
        let right = nextLess[i] - i
        result = (result + arr[i] * left * right) % mod
    }
    
    return result
}

print(sumSubarrayMins([3, 1, 2, 4]))  // 17
```

#### Problem 8: 132 Pattern

```swift
// Find if there exists i < j < k such that arr[i] < arr[k] < arr[j]
// [1, 2, 3, 4] → false
// [3, 1, 4, 2] → true (1 < 2 < 4)
// [-1, 3, 2, 0] → true (-1 < 0 < 3)

func find132pattern(_ nums: [Int]) -> Bool {
    guard nums.count >= 3 else { return false }
    
    var stack = [Int]()
    var second = Int.min
    
    // Traverse from right to left
    for i in stride(from: nums.count - 1, through: 0, by: -1) {
        if nums[i] < second {
            return true  // Found 132 pattern
        }
        
        // Update second with popped elements
        while !stack.isEmpty && nums[i] > stack.last! {
            second = stack.removeLast()
        }
        
        stack.append(nums[i])
    }
    
    return false
}

print(find132pattern([1, 2, 3, 4]))   // false
print(find132pattern([3, 1, 4, 2]))   // true
print(find132pattern([-1, 3, 2, 0]))  // true
```

#### Problem 9: Online Stock Span

```swift
// Calculate stock span - how many consecutive days price <= current
// Prices: [100, 80, 60, 70, 60, 75, 85]
// Spans:  [1, 1, 1, 2, 1, 4, 6]

class StockSpanner {
    private var stack: [(price: Int, span: Int)] = []
    
    func next(_ price: Int) -> Int {
        var span = 1
        
        // Pop smaller prices and add their spans
        while !stack.isEmpty && stack.last!.price <= price {
            span += stack.removeLast().span
        }
        
        stack.append((price, span))
        return span
    }
}

let spanner = StockSpanner()
print(spanner.next(100))  // 1
print(spanner.next(80))   // 1
print(spanner.next(60))   // 1
print(spanner.next(70))   // 2
print(spanner.next(60))   // 1
print(spanner.next(75))   // 4
print(spanner.next(85))   // 6
```

#### Problem 10: Remove Duplicate Letters

```swift
// Remove duplicate letters so result is smallest in lexicographical order
// "bcabc" → "abc"
// "cbacdcbc" → "acdb"

func removeDuplicateLetters(_ s: String) -> String {
    var lastIndex = [Character: Int]()
    var seen = Set<Character>()
    var stack = [Character]()
    
    // Record last index of each character
    for (i, char) in s.enumerated() {
        lastIndex[char] = i
    }
    
    for (i, char) in s.enumerated() {
        if seen.contains(char) {
            continue
        }
        
        // Remove larger characters if they appear later
        while !stack.isEmpty && stack.last! > char && lastIndex[stack.last!]! > i {
            seen.remove(stack.removeLast())
        }
        
        stack.append(char)
        seen.insert(char)
    }
    
    return String(stack)
}

print(removeDuplicateLetters("bcabc"))     // "abc"
print(removeDuplicateLetters("cbacdcbc"))  // "acdb"
```

---

### Monotonic Queue

A queue where elements are always in increasing or decreasing order. Used for sliding window problems.

#### Problem 1: Sliding Window Maximum

```swift
// Find maximum in each window of size k
// [1, 3, -1, -3, 5, 3, 6, 7], k = 3 → [3, 3, 5, 5, 6, 7]

func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    var result = [Int]()
    var deque = [Int]()  // Monotonic decreasing (stores indices)
    
    for i in 0..<nums.count {
        // Remove indices outside window
        while !deque.isEmpty && deque.first! <= i - k {
            deque.removeFirst()
        }
        
        // Remove smaller elements (they'll never be max)
        while !deque.isEmpty && nums[deque.last!] < nums[i] {
            deque.removeLast()
        }
        
        deque.append(i)
        
        // Add max to result
        if i >= k - 1 {
            result.append(nums[deque.first!])
        }
    }
    
    return result
}
```

#### Problem 2: Shortest Subarray with Sum at Least K

```swift
// Find shortest subarray with sum >= k
// [1], k = 1 → 1
// [1, 2], k = 4 → -1

func shortestSubarray(_ nums: [Int], _ k: Int) -> Int {
    let n = nums.count
    var prefixSum = Array(repeating: 0, count: n + 1)
    
    // Calculate prefix sum
    for i in 0..<n {
        prefixSum[i + 1] = prefixSum[i] + nums[i]
    }
    
    var deque = [Int]()
    var minLength = Int.max
    
    for i in 0...n {
        // Check if we can form subarray
        while !deque.isEmpty && prefixSum[i] - prefixSum[deque.first!] >= k {
            minLength = min(minLength, i - deque.removeFirst())
        }
        
        // Maintain increasing deque
        while !deque.isEmpty && prefixSum[i] <= prefixSum[deque.last!] {
            deque.removeLast()
        }
        
        deque.append(i)
    }
    
    return minLength == Int.max ? -1 : minLength
}

print(shortestSubarray([1], 1))           // 1
print(shortestSubarray([1, 2], 4))        // -1
print(shortestSubarray([2, -1, 2], 3))    // 3
```

#### Problem 3: Constrained Subsequence Sum

```swift
// Maximum sum of non-empty subsequence with constraint:
// For each i, arr[i] must be from arr[j] where i - k <= j < i
// [10, 2, -10, 5, 20], k = 2 → 37 (10 + 2 + 5 + 20)

func constrainedSubsetSum(_ nums: [Int], _ k: Int) -> Int {
    var dp = nums
    var deque = [Int]()  // Monotonic decreasing (stores indices)
    var maxSum = nums[0]
    
    for i in 0..<nums.count {
        // Remove indices outside window
        while !deque.isEmpty && deque.first! < i - k {
            deque.removeFirst()
        }
        
        // Calculate dp[i]
        if !deque.isEmpty {
            dp[i] = max(dp[i], nums[i] + dp[deque.first!])
        }
        
        // Remove smaller values
        while !deque.isEmpty && dp[deque.last!] <= dp[i] {
            deque.removeLast()
        }
        
        deque.append(i)
        maxSum = max(maxSum, dp[i])
    }
    
    return maxSum
}

print(constrainedSubsetSum([10, 2, -10, 5, 20], 2))  // 37
```

---

### Monotonic Stack/Queue Patterns

| Problem Type | Stack/Queue | Order | Key Insight |
|--------------|-------------|-------|-------------|
| Next Greater | Stack | Decreasing | Pop when current > top |
| Next Smaller | Stack | Increasing | Pop when current < top |
| Sliding Window Max | Queue (Deque) | Decreasing | Front is always max |
| Sliding Window Min | Queue (Deque) | Increasing | Front is always min |
| Histogram Problems | Stack | Increasing | Calculate area on pop |
| Stock Span | Stack | Decreasing | Count consecutive smaller |

### Decision Tree: Which to Use?

```
Need to find next/previous greater/smaller?
├─ YES → Monotonic Stack
│   ├─ Greater? → Decreasing stack
│   └─ Smaller? → Increasing stack
│
└─ Need sliding window max/min?
    └─ YES → Monotonic Queue (Deque)
        ├─ Maximum? → Decreasing deque
        └─ Minimum? → Increasing deque
```

### Tips & Tricks

1. **Store indices, not values**: Helps track positions and distances
2. **Increasing vs Decreasing**: 
   - Increasing: Pop if current < top (finds smaller)
   - Decreasing: Pop if current > top (finds greater)
3. **Add sentinel**: Helps process remaining stack elements
4. **Left + Right bounds**: Two passes for previous and next
5. **Circular arrays**: Loop twice, use modulo
6. **Deque for windows**: Add to back, remove from front
7. **Calculate on pop**: Often the answer is in what you pop

### Common Mistakes

❌ Storing values instead of indices  
❌ Wrong monotonic order (increasing vs decreasing)  
❌ Forgetting to handle remaining elements in stack  
❌ Not checking isEmpty before accessing last/first  
❌ Wrong boundary conditions in loops  

---

<a name="chapter-14"></a>
## Chapter 14: Hash Tables Deep Dive

Hash tables (dictionaries in Swift) provide **O(1)** average-case lookup, insert, and delete operations. They're one of the most important data structures!

### Hash Table Basics

```swift
// Declaration
var dict = [String: Int]()
var dict2: [Int: String] = [:]

// Time Complexities (Average Case)
// Insert:  O(1)
// Delete:  O(1)
// Search:  O(1)
// Space:   O(n)

// Worst case (hash collisions): O(n)
```

### Hash Table Operations

```swift
var scores = [String: Int]()

// Insert/Update
scores["Alice"] = 95
scores["Bob"] = 87
scores.updateValue(92, forKey: "Charlie")

// Access
let aliceScore = scores["Alice"]        // Optional(95)
let defaultScore = scores["Dave", default: 0]  // 0

// Remove
scores.removeValue(forKey: "Bob")
scores["Charlie"] = nil

// Check existence
if scores.keys.contains("Alice") { }
if scores["Alice"] != nil { }

// Iteration
for (name, score) in scores {
    print("\(name): \(score)")
}

for name in scores.keys {
    print(name)
}

for score in scores.values {
    print(score)
}
```

---

### Classic Hash Table Problems

#### Problem 1: Two Sum

```swift
// Find indices where nums[i] + nums[j] = target
// [2, 7, 11, 15], target = 9 → [0, 1]

func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var map = [Int: Int]()  // value: index
    
    for (i, num) in nums.enumerated() {
        let complement = target - num
        
        if let j = map[complement] {
            return [j, i]
        }
        
        map[num] = i
    }
    
    return []
}

print(twoSum([2, 7, 11, 15], 9))  // [0, 1]
```

#### Problem 2: Group Anagrams

```swift
// Group strings that are anagrams
// ["eat", "tea", "tan", "ate", "nat", "bat"]
// → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]

func groupAnagrams(_ strs: [String]) -> [[String]] {
    var groups = [String: [String]]()
    
    for str in strs {
        let key = String(str.sorted())
        groups[key, default: []].append(str)
    }
    
    return Array(groups.values)
}

// Alternative: Use character count as key
func groupAnagrams2(_ strs: [String]) -> [[String]] {
    var groups = [[Int]: [String]]()
    
    for str in strs {
        var count = Array(repeating: 0, count: 26)
        
        for char in str {
            let index = Int(char.asciiValue! - Character("a").asciiValue!)
            count[index] += 1
        }
        
        groups[count, default: []].append(str)
    }
    
    return Array(groups.values)
}

print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
```

#### Problem 3: Longest Consecutive Sequence

```swift
// Find length of longest consecutive sequence
// [100, 4, 200, 1, 3, 2] → 4 (sequence: 1, 2, 3, 4)

func longestConsecutive(_ nums: [Int]) -> Int {
    let numSet = Set(nums)
    var maxLength = 0
    
    for num in numSet {
        // Only start counting if it's the beginning of sequence
        if !numSet.contains(num - 1) {
            var currentNum = num
            var currentLength = 1
            
            while numSet.contains(currentNum + 1) {
                currentNum += 1
                currentLength += 1
            }
            
            maxLength = max(maxLength, currentLength)
        }
    }
    
    return maxLength
}

print(longestConsecutive([100, 4, 200, 1, 3, 2]))  // 4
```

#### Problem 4: Subarray Sum Equals K

```swift
// Count subarrays with sum = k
// [1, 1, 1], k = 2 → 2

func subarraySum(_ nums: [Int], _ k: Int) -> Int {
    var count = 0
    var sum = 0
    var sumCount = [Int: Int]()
    sumCount[0] = 1  // Important: empty subarray
    
    for num in nums {
        sum += num
        
        // If (sum - k) exists, we found subarrays
        if let prevCount = sumCount[sum - k] {
            count += prevCount
        }
        
        sumCount[sum, default: 0] += 1
    }
    
    return count
}

print(subarraySum([1, 1, 1], 2))       // 2
print(subarraySum([1, 2, 3], 3))       // 2
print(subarraySum([1, -1, 0], 0))      // 3
```

#### Problem 5: Longest Substring Without Repeating Characters

```swift
// Find length of longest substring without repeating chars
// "abcabcbb" → 3 ("abc")

func lengthOfLongestSubstring(_ s: String) -> Int {
    let chars = Array(s)
    var lastIndex = [Character: Int]()
    var maxLength = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        if let prevIndex = lastIndex[char], prevIndex >= start {
            start = prevIndex + 1
        }
        
        lastIndex[char] = end
        maxLength = max(maxLength, end - start + 1)
    }
    
    return maxLength
}

print(lengthOfLongestSubstring("abcabcbb"))  // 3
print(lengthOfLongestSubstring("bbbbb"))     // 1
print(lengthOfLongestSubstring("pwwkew"))    // 3
```

#### Problem 6: Top K Frequent Elements

```swift
// Find k most frequent elements
// [1, 1, 1, 2, 2, 3], k = 2 → [1, 2]

func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
    var freq = [Int: Int]()
    
    // Count frequencies
    for num in nums {
        freq[num, default: 0] += 1
    }
    
    // Sort by frequency
    let sorted = freq.sorted { $0.value > $1.value }
    
    return Array(sorted.prefix(k).map { $0.key })
}

// Using bucket sort - O(n)
func topKFrequent2(_ nums: [Int], _ k: Int) -> [Int] {
    var freq = [Int: Int]()
    
    for num in nums {
        freq[num, default: 0] += 1
    }
    
    // Bucket[i] contains numbers with frequency i
    var buckets = Array(repeating: [Int](), count: nums.count + 1)
    
    for (num, count) in freq {
        buckets[count].append(num)
    }
    
    // Collect k most frequent
    var result = [Int]()
    for i in stride(from: buckets.count - 1, through: 0, by: -1) {
        result.append(contentsOf: buckets[i])
        if result.count >= k {
            break
        }
    }
    
    return Array(result.prefix(k))
}

print(topKFrequent([1, 1, 1, 2, 2, 3], 2))  // [1, 2]
```

#### Problem 7: 4Sum II

```swift
// Count tuples (i, j, k, l) where nums1[i] + nums2[j] + nums3[k] + nums4[l] = 0
// All arrays have same length n

func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
    var sumCount = [Int: Int]()
    
    // Store all possible sums of nums1 and nums2
    for num1 in nums1 {
        for num2 in nums2 {
            let sum = num1 + num2
            sumCount[sum, default: 0] += 1
        }
    }
    
    var count = 0
    
    // Check if negative sum exists in map
    for num3 in nums3 {
        for num4 in nums4 {
            let target = -(num3 + num4)
            count += sumCount[target, default: 0]
        }
    }
    
    return count
}

print(fourSumCount([1, 2], [-2, -1], [-1, 2], [0, 2]))  // 2
```

#### Problem 8: Isomorphic Strings

```swift
// Check if two strings are isomorphic
// "egg" and "add" → true
// "foo" and "bar" → false

func isIsomorphic(_ s: String, _ t: String) -> Bool {
    guard s.count == t.count else { return false }
    
    let sChars = Array(s)
    let tChars = Array(t)
    
    var sToT = [Character: Character]()
    var tToS = [Character: Character]()
    
    for i in 0..<sChars.count {
        let sChar = sChars[i]
        let tChar = tChars[i]
        
        if let mapped = sToT[sChar] {
            if mapped != tChar {
                return false
            }
        } else {
            sToT[sChar] = tChar
        }
        
        if let mapped = tToS[tChar] {
            if mapped != sChar {
                return false
            }
        } else {
            tToS[tChar] = sChar
        }
    }
    
    return true
}

print(isIsomorphic("egg", "add"))   // true
print(isIsomorphic("foo", "bar"))   // false
print(isIsomorphic("paper", "title"))  // true
```

#### Problem 9: Happy Number

```swift
// Number is happy if eventually reaches 1 by repeatedly:
// Replace number by sum of squares of its digits
// 19 → 1² + 9² = 82 → 8² + 2² = 68 → 6² + 8² = 100 → 1² + 0² + 0² = 1

func isHappy(_ n: Int) -> Bool {
    var seen = Set<Int>()
    var current = n
    
    while current != 1 && !seen.contains(current) {
        seen.insert(current)
        current = sumOfSquares(current)
    }
    
    return current == 1
    
    func sumOfSquares(_ num: Int) -> Int {
        var sum = 0
        var n = num
        
        while n > 0 {
            let digit = n % 10
            sum += digit * digit
            n /= 10
        }
        
        return sum
    }
}

print(isHappy(19))  // true
print(isHappy(2))   // false
```

#### Problem 10: LRU Cache (Hash Map + Doubly Linked List)

```swift
// Implemented earlier, but here's the key insight:
// Hash map for O(1) lookup
// Doubly linked list for O(1) insertion/deletion

class LRUCache {
    class Node {
        var key: Int
        var value: Int
        var prev: Node?
        var next: Node?
        
        init(_ key: Int, _ value: Int) {
            self.key = key
            self.value = value
        }
    }
    
    private var capacity: Int
    private var cache: [Int: Node] = [:]
    private var head: Node
    private var tail: Node
    
    init(_ capacity: Int) {
        self.capacity = capacity
        head = Node(0, 0)
        tail = Node(0, 0)
        head.next = tail
        tail.prev = head
    }
    
    func get(_ key: Int) -> Int {
        guard let node = cache[key] else { return -1 }
        moveToFront(node)
        return node.value
    }
    
    func put(_ key: Int, _ value: Int) {
        if let node = cache[key] {
            node.value = value
            moveToFront(node)
        } else {
            let newNode = Node(key, value)
            cache[key] = newNode
            addToFront(newNode)
            
            if cache.count > capacity {
                if let lru = tail.prev {
                    removeNode(lru)
                    cache.removeValue(forKey: lru.key)
                }
            }
        }
    }
    
    private func addToFront(_ node: Node) {
        node.next = head.next
        node.prev = head
        head.next?.prev = node
        head.next = node
    }
    
    private func removeNode(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func moveToFront(_ node: Node) {
        removeNode(node)
        addToFront(node)
    }
}
```

---

### Advanced Hash Table Techniques

#### Technique 1: Prefix Sum + Hash Map

```swift
// Find subarray with sum = k
func subarraySum(_ nums: [Int], _ k: Int) -> Int {
    var count = 0
    var sum = 0
    var sumFreq = [0: 1]  // sum: frequency
    
    for num in nums {
        sum += num
        count += sumFreq[sum - k, default: 0]
        sumFreq[sum, default: 0] += 1
    }
    
    return count
}
```

#### Technique 2: Frequency Counter Pattern

```swift
// Check if array1 is subset of array2
func isSubset(_ arr1: [Int], _ arr2: [Int]) -> Bool {
    var freq = [Int: Int]()
    
    for num in arr2 {
        freq[num, default: 0] += 1
    }
    
    for num in arr1 {
        guard let count = freq[num], count > 0 else {
            return false
        }
        freq[num] = count - 1
    }
    
    return true
}
```

#### Technique 3: Index Mapping

```swift
// Find indices of two numbers that sum to target
func twoSumIndices(_ nums: [Int], _ target: Int) -> [Int] {
    var indexMap = [Int: Int]()
    
    for (i, num) in nums.enumerated() {
        if let j = indexMap[target - num] {
            return [j, i]
        }
        indexMap[num] = i
    }
    
    return []
}
```

#### Technique 4: Sliding Window + Hash Map

```swift
// Longest substring with at most k distinct characters
func lengthOfLongestSubstringKDistinct(_ s: String, _ k: Int) -> Int {
    let chars = Array(s)
    var freq = [Character: Int]()
    var maxLen = 0
    var start = 0
    
    for (end, char) in chars.enumerated() {
        freq[char, default: 0] += 1
        
        while freq.count > k {
            let leftChar = chars[start]
            freq[leftChar]! -= 1
            if freq[leftChar] == 0 {
                freq.removeValue(forKey: leftChar)
            }
            start += 1
        }
        
        maxLen = max(maxLen, end - start + 1)
    }
    
    return maxLen
}
```

---

### Hash Table Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Value → Index** | Find pairs, duplicates | Two sum |
| **Frequency Counter** | Anagrams, k frequent | Group anagrams |
| **Prefix Sum** | Subarray sum | Subarray sum = k |
| **Character Mapping** | Isomorphic, patterns | Isomorphic strings |
| **Visited Set** | Cycle detection | Happy number |
| **Sliding Window** | Substring problems | Longest substring |

### Tips & Tricks

1. **Default values**: Use `dict[key, default: 0]` to avoid nil checks
2. **Set for existence**: Use `Set` when you only need to check existence
3. **Count then process**: Often need two passes (count, then use counts)
4. **Complement pattern**: Store value, search for complement
5. **Prefix sums**: Transform array problems to hash map lookups
6. **Multiple maps**: Don't hesitate to use 2+ hash maps
7. **Keys can be complex**: Arrays, tuples as keys (must be Hashable)

### When NOT to Use Hash Tables

❌ **Need ordering**: Hash tables don't maintain order  
❌ **Range queries**: Can't efficiently query ranges  
❌ **Memory constrained**: Hash tables use extra space  
❌ **Small datasets**: Overhead might not be worth it  

### Hash Collisions

```swift
// Swift handles collisions internally
// Load factor: numberOfElements / capacity
// Rehashing happens automatically when load factor is high

// For custom hash:
struct Point: Hashable {
    let x: Int
    let y: Int
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(x)
        hasher.combine(y)
    }
}

var pointSet = Set<Point>()
pointSet.insert(Point(x: 1, y: 2))
```

---

**🎯 Practice Problems:**
1. Minimum Window Substring
2. Fraction to Recurring Decimal
3. Contains Duplicate II & III
4. Valid Sudoku
5. Longest Substring with At Most Two Distinct Characters

<a name="chapter-15"></a>
## Chapter 15: Set Operations

Sets are collections of **unique** elements with no defined order. They excel at membership testing and mathematical set operations!

### Set Basics in Swift

```swift
// Declaration
var set1: Set<Int> = [1, 2, 3]
var set2 = Set([1, 2, 2, 3])  // [1, 2, 3] - duplicates removed

// Time Complexities (Average Case)
// Insert:   O(1)
// Delete:   O(1)
// Search:   O(1)
// Contains: O(1)
```

### Basic Set Operations

```swift
var fruits: Set<String> = ["apple", "banana", "orange"]

// Insert
fruits.insert("mango")

// Remove
fruits.remove("banana")

// Contains (fastest way to check membership)
if fruits.contains("apple") {
    print("Found apple")
}

// Count & isEmpty
print(fruits.count)      // 3
print(fruits.isEmpty)    // false

// Iteration (unordered!)
for fruit in fruits {
    print(fruit)
}

// Convert to sorted array
let sortedFruits = fruits.sorted()
```

---

### Mathematical Set Operations

```swift
let set1: Set = [1, 2, 3, 4, 5]
let set2: Set = [4, 5, 6, 7, 8]

// Union (all elements from both sets)
let union = set1.union(set2)
// {1, 2, 3, 4, 5, 6, 7, 8}

// Intersection (common elements)
let intersection = set1.intersection(set2)
// {4, 5}

// Difference (in set1 but not in set2)
let difference = set1.subtracting(set2)
// {1, 2, 3}

// Symmetric Difference (in either, but not both)
let symmetricDiff = set1.symmetricDifference(set2)
// {1, 2, 3, 6, 7, 8}

// Subset check
let subset: Set = [2, 3]
print(subset.isSubset(of: set1))        // true
print(set1.isSuperset(of: subset))      // true

// Disjoint check (no common elements)
let set3: Set = [10, 11, 12]
print(set1.isDisjoint(with: set3))      // true
```

---

### Classic Set Problems

#### Problem 1: Contains Duplicate

```swift
// Check if array contains duplicates
// [1, 2, 3, 1] → true
// [1, 2, 3, 4] → false

func containsDuplicate(_ nums: [Int]) -> Bool {
    var seen = Set<Int>()
    
    for num in nums {
        if seen.contains(num) {
            return true
        }
        seen.insert(num)
    }
    
    return false
}

// Shorter version
func containsDuplicate2(_ nums: [Int]) -> Bool {
    return Set(nums).count != nums.count
}

print(containsDuplicate([1, 2, 3, 1]))  // true
print(containsDuplicate([1, 2, 3, 4]))  // false
```

#### Problem 2: Intersection of Two Arrays

```swift
// Find intersection of two arrays
// [1, 2, 2, 1], [2, 2] → [2]

func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
    let set1 = Set(nums1)
    let set2 = Set(nums2)
    return Array(set1.intersection(set2))
}

// Or manually
func intersection2(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
    let set1 = Set(nums1)
    var result = Set<Int>()
    
    for num in nums2 {
        if set1.contains(num) {
            result.insert(num)
        }
    }
    
    return Array(result)
}

print(intersection([1, 2, 2, 1], [2, 2]))  // [2]
print(intersection([4, 9, 5], [9, 4, 9, 8, 4]))  // [9, 4]
```

#### Problem 3: Intersection of Two Arrays II (with duplicates)

```swift
// Include duplicates in result
// [1, 2, 2, 1], [2, 2] → [2, 2]

func intersect(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
    var freq = [Int: Int]()
    var result = [Int]()
    
    // Count elements in nums1
    for num in nums1 {
        freq[num, default: 0] += 1
    }
    
    // Find matches in nums2
    for num in nums2 {
        if let count = freq[num], count > 0 {
            result.append(num)
            freq[num] = count - 1
        }
    }
    
    return result
}

print(intersect([1, 2, 2, 1], [2, 2]))  // [2, 2]
print(intersect([4, 9, 5], [9, 4, 9, 8, 4]))  // [4, 9]
```

#### Problem 4: Single Number

```swift
// Every element appears twice except one
// [2, 2, 1] → 1
// [4, 1, 2, 1, 2] → 4

// Using Set (O(n) space)
func singleNumber(_ nums: [Int]) -> Int {
    var seen = Set<Int>()
    
    for num in nums {
        if seen.contains(num) {
            seen.remove(num)
        } else {
            seen.insert(num)
        }
    }
    
    return seen.first!
}

// Using XOR (O(1) space) - Better!
func singleNumber2(_ nums: [Int]) -> Int {
    var result = 0
    for num in nums {
        result ^= num  // XOR: a ^ a = 0, a ^ 0 = a
    }
    return result
}

print(singleNumber([2, 2, 1]))        // 1
print(singleNumber([4, 1, 2, 1, 2]))  // 4
```

#### Problem 5: Happy Number (Using Set for Cycle Detection)

```swift
// Already covered, but demonstrates Set for cycle detection
func isHappy(_ n: Int) -> Bool {
    var seen = Set<Int>()
    var current = n
    
    while current != 1 {
        if seen.contains(current) {
            return false  // Cycle detected
        }
        seen.insert(current)
        current = sumOfSquares(current)
    }
    
    return true
    
    func sumOfSquares(_ num: Int) -> Int {
        var sum = 0
        var n = num
        while n > 0 {
            let digit = n % 10
            sum += digit * digit
            n /= 10
        }
        return sum
    }
}
```

#### Problem 6: Longest Consecutive Sequence

```swift
// Find longest consecutive sequence
// [100, 4, 200, 1, 3, 2] → 4 (sequence: 1, 2, 3, 4)

func longestConsecutive(_ nums: [Int]) -> Int {
    let numSet = Set(nums)
    var maxLength = 0
    
    for num in numSet {
        // Only count from sequence start
        if !numSet.contains(num - 1) {
            var currentNum = num
            var currentLength = 1
            
            while numSet.contains(currentNum + 1) {
                currentNum += 1
                currentLength += 1
            }
            
            maxLength = max(maxLength, currentLength)
        }
    }
    
    return maxLength
}

print(longestConsecutive([100, 4, 200, 1, 3, 2]))  // 4
```

#### Problem 7: Distribute Candies

```swift
// Alice has n candies, wants to give half to brother
// Maximize variety of candies Alice keeps
// [1, 1, 2, 2, 3, 3] → 3
// [1, 1, 2, 3] → 2

func distributeCandies(_ candyType: [Int]) -> Int {
    let uniqueTypes = Set(candyType).count
    let maxCandies = candyType.count / 2
    return min(uniqueTypes, maxCandies)
}

print(distributeCandies([1, 1, 2, 2, 3, 3]))  // 3
print(distributeCandies([1, 1, 2, 3]))        // 2
```

#### Problem 8: Find All Duplicates

```swift
// Find all elements that appear twice
// [4, 3, 2, 7, 8, 2, 3, 1] → [2, 3]

func findDuplicates(_ nums: [Int]) -> [Int] {
    var seen = Set<Int>()
    var duplicates = Set<Int>()
    
    for num in nums {
        if seen.contains(num) {
            duplicates.insert(num)
        } else {
            seen.insert(num)
        }
    }
    
    return Array(duplicates)
}

print(findDuplicates([4, 3, 2, 7, 8, 2, 3, 1]))  // [2, 3]
```

#### Problem 9: Jewels and Stones

```swift
// Count how many stones are jewels
// jewels = "aA", stones = "aAAbbbb" → 3

func numJewelsInStones(_ jewels: String, _ stones: String) -> Int {
    let jewelSet = Set(jewels)
    var count = 0
    
    for stone in stones {
        if jewelSet.contains(stone) {
            count += 1
        }
    }
    
    return count
}

// Functional approach
func numJewelsInStones2(_ jewels: String, _ stones: String) -> Int {
    let jewelSet = Set(jewels)
    return stones.filter { jewelSet.contains($0) }.count
}

print(numJewelsInStones("aA", "aAAbbbb"))  // 3
```

#### Problem 10: Word Pattern

```swift
// Check if pattern matches string
// pattern = "abba", s = "dog cat cat dog" → true
// pattern = "abba", s = "dog cat cat fish" → false

func wordPattern(_ pattern: String, _ s: String) -> Bool {
    let words = s.split(separator: " ").map(String.init)
    let chars = Array(pattern)
    
    guard chars.count == words.count else { return false }
    
    var charToWord = [Character: String]()
    var wordSet = Set<String>()
    
    for i in 0..<chars.count {
        let char = chars[i]
        let word = words[i]
        
        if let mapped = charToWord[char] {
            if mapped != word {
                return false
            }
        } else {
            // Check if word is already mapped to another char
            if wordSet.contains(word) {
                return false
            }
            charToWord[char] = word
            wordSet.insert(word)
        }
    }
    
    return true
}

print(wordPattern("abba", "dog cat cat dog"))   // true
print(wordPattern("abba", "dog cat cat fish"))  // false
```

---

### Set-Based Algorithms

#### Algorithm 1: Finding Missing and Duplicate Numbers

```swift
// Array should contain 1 to n, find duplicate and missing
// [1, 2, 2, 4] → duplicate: 2, missing: 3

func findErrorNums(_ nums: [Int]) -> [Int] {
    let numSet = Set(nums)
    var duplicate = 0
    var missing = 0
    
    // Find duplicate
    for num in nums {
        if nums.filter({ $0 == num }).count > 1 {
            duplicate = num
            break
        }
    }
    
    // Find missing
    for i in 1...nums.count {
        if !numSet.contains(i) {
            missing = i
            break
        }
    }
    
    return [duplicate, missing]
}

// More efficient version
func findErrorNums2(_ nums: [Int]) -> [Int] {
    var seen = Set<Int>()
    var duplicate = 0
    
    for num in nums {
        if seen.contains(num) {
            duplicate = num
        }
        seen.insert(num)
    }
    
    var missing = 0
    for i in 1...nums.count {
        if !seen.contains(i) {
            missing = i
            break
        }
    }
    
    return [duplicate, missing]
}
```

#### Algorithm 2: First Missing Positive

```swift
// Find smallest missing positive integer
// [1, 2, 0] → 3
// [3, 4, -1, 1] → 2
// [7, 8, 9, 11, 12] → 1

func firstMissingPositive(_ nums: [Int]) -> Int {
    let positives = Set(nums.filter { $0 > 0 })
    
    var i = 1
    while positives.contains(i) {
        i += 1
    }
    
    return i
}

print(firstMissingPositive([1, 2, 0]))         // 3
print(firstMissingPositive([3, 4, -1, 1]))     // 2
print(firstMissingPositive([7, 8, 9, 11, 12])) // 1
```

#### Algorithm 3: Unique Email Addresses

```swift
// Count unique email addresses
// Local name: before @, can have . (ignored) or + (ignore after)
// Domain name: after @
// ["test.email+alex@leetcode.com", "test.e.mail+bob.cathy@leetcode.com"]
// → 2 (both resolve to testemail@leetcode.com)

func numUniqueEmails(_ emails: [String]) -> Int {
    var uniqueEmails = Set<String>()
    
    for email in emails {
        let parts = email.split(separator: "@")
        guard parts.count == 2 else { continue }
        
        var local = String(parts[0])
        let domain = String(parts[1])
        
        // Remove dots
        local = local.replacingOccurrences(of: ".", with: "")
        
        // Remove everything after +
        if let plusIndex = local.firstIndex(of: "+") {
            local = String(local[..<plusIndex])
        }
        
        uniqueEmails.insert(local + "@" + domain)
    }
    
    return uniqueEmails.count
}

let emails = ["test.email+alex@leetcode.com", 
              "test.e.mail+bob.cathy@leetcode.com",
              "testemail+david@lee.tcode.com"]
print(numUniqueEmails(emails))  // 2
```

---

### Set vs Array vs Dictionary

| Feature | Array | Set | Dictionary |
|---------|-------|-----|------------|
| Order | ✅ Preserved | ❌ Unordered | ❌ Unordered |
| Duplicates | ✅ Allowed | ❌ Unique only | Keys unique |
| Index Access | O(1) | ❌ N/A | O(1) by key |
| Contains | O(n) | O(1) | O(1) |
| Insert | O(1) end | O(1) | O(1) |
| Best For | Sequential | Membership | Key-value |

### When to Use Sets

✅ **Use Set when:**
- Need to check membership frequently
- Want to remove duplicates
- Performing set operations (union, intersection)
- Order doesn't matter
- Need unique elements guarantee

❌ **Don't use Set when:**
- Need to maintain order
- Need to access by index
- Need to store duplicates
- Need key-value pairs (use Dictionary)

### Set Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Deduplication** | Remove duplicates | Unique elements |
| **Membership Testing** | Check existence | Contains duplicate |
| **Set Operations** | Union, intersection | Find common elements |
| **Cycle Detection** | Detect loops | Happy number |
| **Tracking Seen** | Mark visited | Graph traversal |
| **Complement Search** | Find missing | First missing positive |

### Tips & Tricks

1. **Quick duplicate check**: `Set(array).count != array.count`
2. **Set comprehension**: Use `filter`, `map` with sets
3. **Multiple sets**: Don't hesitate to use multiple sets
4. **Set vs Dictionary**: If you only need keys, use Set
5. **Sorted output**: Convert to array and sort when needed
6. **Set intersection**: Often faster than nested loops
7. **Custom types**: Make Hashable for custom objects in sets

```swift
// Custom type in Set
struct Point: Hashable {
    let x: Int
    let y: Int
}

var points: Set<Point> = []
points.insert(Point(x: 1, y: 2))
print(points.contains(Point(x: 1, y: 2)))  // true
```

---

<a name="chapter-16"></a>
## Chapter 16: Binary Trees

Binary trees are hierarchical data structures where each node has at most two children. They're fundamental to many algorithms!

### Binary Tree Basics

```swift
// Node definition
class TreeNode {
    var val: Int
    var left: TreeNode?
    var right: TreeNode?
    
    init(_ val: Int) {
        self.val = val
        self.left = nil
        self.right = nil
    }
}

// Creating a tree:
//       1
//      / \
//     2   3
//    / \
//   4   5

let root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left?.left = TreeNode(4)
root.left?.right = TreeNode(5)
```

### Tree Terminology

```
       1       ← Root
      / \
     2   3     ← Level 1 (internal nodes)
    / \   \
   4   5   6   ← Level 2 (leaves)

- Root: Top node (1)
- Parent: Node with children (1, 2, 3)
- Child: Node with parent (2, 3, 4, 5, 6)
- Leaf: Node with no children (4, 5, 6)
- Siblings: Share same parent (2, 3)
- Height: Longest path from node to leaf
- Depth: Distance from root to node
```

### Tree Properties

```swift
// Height of tree
func height(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    return 1 + max(height(root.left), height(root.right))
}

// Count nodes
func countNodes(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    return 1 + countNodes(root.left) + countNodes(root.right)
}

// Check if balanced
func isBalanced(_ root: TreeNode?) -> Bool {
    return checkBalance(root).isBalanced
    
    func checkBalance(_ node: TreeNode?) -> (height: Int, isBalanced: Bool) {
        guard let node = node else {
            return (0, true)
        }
        
        let left = checkBalance(node.left)
        let right = checkBalance(node.right)
        
        let balanced = left.isBalanced && 
                      right.isBalanced && 
                      abs(left.height - right.height) <= 1
        
        return (1 + max(left.height, right.height), balanced)
    }
}
```

---

### Tree Traversals

#### 1. Inorder Traversal (Left → Root → Right)

```swift
// Recursive
func inorderTraversal(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    result += inorderTraversal(root.left)
    result.append(root.val)
    result += inorderTraversal(root.right)
    
    return result
}

// Iterative (using stack)
func inorderTraversalIterative(_ root: TreeNode?) -> [Int] {
    var result = [Int]()
    var stack = [TreeNode]()
    var current = root
    
    while current != nil || !stack.isEmpty {
        // Go to leftmost node
        while current != nil {
            stack.append(current!)
            current = current?.left
        }
        
        // Process node
        current = stack.removeLast()
        result.append(current!.val)
        
        // Go to right subtree
        current = current?.right
    }
    
    return result
}

// For BST, inorder gives sorted order!
```

#### 2. Preorder Traversal (Root → Left → Right)

```swift
// Recursive
func preorderTraversal(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [root.val]
    result += preorderTraversal(root.left)
    result += preorderTraversal(root.right)
    
    return result
}

// Iterative
func preorderTraversalIterative(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    var stack = [root]
    
    while !stack.isEmpty {
        let node = stack.removeLast()
        result.append(node.val)
        
        // Push right first (so left is processed first)
        if let right = node.right {
            stack.append(right)
        }
        if let left = node.left {
            stack.append(left)
        }
    }
    
    return result
}
```

#### 3. Postorder Traversal (Left → Right → Root)

```swift
// Recursive
func postorderTraversal(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    result += postorderTraversal(root.left)
    result += postorderTraversal(root.right)
    result.append(root.val)
    
    return result
}

// Iterative (two stacks)
func postorderTraversalIterative(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    var stack1 = [root]
    var stack2 = [TreeNode]()
    
    while !stack1.isEmpty {
        let node = stack1.removeLast()
        stack2.append(node)
        
        if let left = node.left {
            stack1.append(left)
        }
        if let right = node.right {
            stack1.append(right)
        }
    }
    
    while !stack2.isEmpty {
        result.append(stack2.removeLast().val)
    }
    
    return result
}
```

#### 4. Level Order Traversal (BFS)

```swift
// Level by level from left to right
func levelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }
    
    var result = [[Int]]()
    var queue = [root]
    
    while !queue.isEmpty {
        let levelSize = queue.count
        var currentLevel = [Int]()
        
        for _ in 0..<levelSize {
            let node = queue.removeFirst()
            currentLevel.append(node.val)
            
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
        
        result.append(currentLevel)
    }
    
    return result
}

// Example output for tree above:
// [[1], [2, 3], [4, 5, 6]]
```

---

### Classic Binary Tree Problems

#### Problem 1: Maximum Depth

```swift
// Find maximum depth (height) of tree
func maxDepth(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
}

// Iterative (level order)
func maxDepthIterative(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    
    var queue = [root]
    var depth = 0
    
    while !queue.isEmpty {
        let levelSize = queue.count
        depth += 1
        
        for _ in 0..<levelSize {
            let node = queue.removeFirst()
            
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
    }
    
    return depth
}
```

#### Problem 2: Minimum Depth

```swift
// Find minimum depth (shortest path to leaf)
func minDepth(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    
    // Leaf node
    if root.left == nil && root.right == nil {
        return 1
    }
    
    // Only right child
    if root.left == nil {
        return 1 + minDepth(root.right)
    }
    
    // Only left child
    if root.right == nil {
        return 1 + minDepth(root.left)
    }
    
    // Both children
    return 1 + min(minDepth(root.left), minDepth(root.right))
}
```

#### Problem 3: Invert Binary Tree

```swift
// Mirror the tree
//     4           4
//    / \         / \
//   2   7  →    7   2
//  / \ / \     / \ / \
// 1  3 6  9   9  6 3  1

func invertTree(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    
    // Swap children
    let temp = root.left
    root.left = root.right
    root.right = temp
    
    // Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)
    
    return root
}
```

#### Problem 4: Same Tree

```swift
// Check if two trees are identical
func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
    // Both nil
    if p == nil && q == nil {
        return true
    }
    
    // One is nil
    if p == nil || q == nil {
        return false
    }
    
    // Compare values and recurse
    return p!.val == q!.val &&
           isSameTree(p!.left, q!.left) &&
           isSameTree(p!.right, q!.right)
}
```

#### Problem 5: Symmetric Tree

```swift
// Check if tree is mirror of itself
//     1
//    / \
//   2   2
//  / \ / \
// 3  4 4  3  → true

func isSymmetric(_ root: TreeNode?) -> Bool {
    return isMirror(root?.left, root?.right)
    
    func isMirror(_ left: TreeNode?, _ right: TreeNode?) -> Bool {
        if left == nil && right == nil {
            return true
        }
        
        if left == nil || right == nil {
            return false
        }
        
        return left!.val == right!.val &&
               isMirror(left!.left, right!.right) &&
               isMirror(left!.right, right!.left)
    }
}
```

#### Problem 6: Path Sum

```swift
// Check if path from root to leaf sums to target
func hasPathSum(_ root: TreeNode?, _ targetSum: Int) -> Bool {
    guard let root = root else { return false }
    
    // Leaf node
    if root.left == nil && root.right == nil {
        return root.val == targetSum
    }
    
    let remaining = targetSum - root.val
    return hasPathSum(root.left, remaining) || 
           hasPathSum(root.right, remaining)
}

// Find all paths that sum to target
func pathSum(_ root: TreeNode?, _ targetSum: Int) -> [[Int]] {
    var result = [[Int]]()
    var path = [Int]()
    
    dfs(root, targetSum, &path, &result)
    return result
    
    func dfs(_ node: TreeNode?, _ remaining: Int, 
             _ path: inout [Int], _ result: inout [[Int]]) {
        guard let node = node else { return }
        
        path.append(node.val)
        
        // Leaf node with correct sum
        if node.left == nil && node.right == nil && node.val == remaining {
            result.append(path)
        }
        
        dfs(node.left, remaining - node.val, &path, &result)
        dfs(node.right, remaining - node.val, &path, &result)
        
        path.removeLast()  // Backtrack
    }
}
```

#### Problem 7: Diameter of Binary Tree

```swift
// Longest path between any two nodes
func diameterOfBinaryTree(_ root: TreeNode?) -> Int {
    var diameter = 0
    
    @discardableResult
    func height(_ node: TreeNode?) -> Int {
        guard let node = node else { return 0 }
        
        let leftHeight = height(node.left)
        let rightHeight = height(node.right)
        
        // Update diameter
        diameter = max(diameter, leftHeight + rightHeight)
        
        return 1 + max(leftHeight, rightHeight)
    }
    
    height(root)
    return diameter
}
```

#### Problem 8: Lowest Common Ancestor

```swift
// Find LCA of two nodes in binary tree
func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    
    // Found one of the nodes
    if root === p || root === q {
        return root
    }
    
    let left = lowestCommonAncestor(root.left, p, q)
    let right = lowestCommonAncestor(root.right, p, q)
    
    // Found in both subtrees - this is LCA
    if left != nil && right != nil {
        return root
    }
    
    // Return whichever is not nil
    return left ?? right
}
```

#### Problem 9: Binary Tree Paths

```swift
// Find all root-to-leaf paths
// Tree: 1
//      / \
//     2   3
//      \
//       5
// Output: ["1->2->5", "1->3"]

func binaryTreePaths(_ root: TreeNode?) -> [String] {
    var result = [String]()
    
    func dfs(_ node: TreeNode?, _ path: String) {
        guard let node = node else { return }
        
        let currentPath = path.isEmpty ? "\(node.val)" : "\(path)->\(node.val)"
        
        // Leaf node
        if node.left == nil && node.right == nil {
            result.append(currentPath)
            return
        }
        
        dfs(node.left, currentPath)
        dfs(node.right, currentPath)
    }
    
    dfs(root, "")
    return result
}
```

#### Problem 10: Flatten Binary Tree to Linked List

```swift
// Flatten to linked list in preorder
//     1              1
//    / \              \
//   2   5      →       2
//  / \   \              \
// 3   4   6              3
//                         \
//                          4
//                           \
//                            5
//                             \
//                              6

func flatten(_ root: TreeNode?) {
    guard let root = root else { return }
    
    flatten(root.left)
    flatten(root.right)
    
    // Save right subtree
    let rightSubtree = root.right
    
    // Move left subtree to right
    root.right = root.left
    root.left = nil
    
    // Attach right subtree to end
    var current = root
    while current.right != nil {
        current = current.right!
    }
    current.right = rightSubtree
}
```

---

### Tree Traversal Summary

| Traversal | Order | Use Case | Recursive? |
|-----------|-------|----------|-----------|
| Inorder | L-Root-R | BST sorted | ✅ |
| Preorder | Root-L-R | Copy tree | ✅ |
| Postorder | L-R-Root | Delete tree | ✅ |
| Level Order | Level by level | BFS, min depth | ❌ |

### Common Tree Patterns

1. **Recursive DFS**: Most tree problems
2. **Level Order (BFS)**: Level-based problems
3. **Path tracking**: Maintain path array
4. **Two pointers**: Compare two trees
5. **Post-order for bottom-up**: Height, diameter
6. **Pre-order for top-down**: Path sum, copying

### Tips & Tricks

1. **Base case**: Always handle nil nodes
2. **Leaf check**: `left == nil && right == nil`
3. **Height vs Depth**: Height from bottom, depth from top
4. **Return values**: Use return values to pass info up
5. **Modify in-place**: Can modify tree during traversal
6. **Helper functions**: Use inner functions for extra parameters
7. **Draw it**: Visualize recursion on paper

---

**🎯 Practice Problems:**
1. Construct Binary Tree from Preorder and Inorder
2. Serialize and Deserialize Binary Tree
3. Binary Tree Right Side View
4. Count Good Nodes in Binary Tree
5. Sum Root to Leaf Numbers

<a name="chapter-17"></a>
## Chapter 17: Binary Search Trees (BST)

A Binary Search Tree is a binary tree with a special property: for every node, all values in the **left subtree** are **smaller**, and all values in the **right subtree** are **larger**.

### BST Properties

```
       8
      / \
     3   10
    / \    \
   1   6   14
      / \  /
     4  7 13

BST Property:
- Left subtree of 8: {3, 1, 6, 4, 7} - all < 8
- Right subtree of 8: {10, 14, 13} - all > 8
- This applies recursively to all nodes!
```

### BST Time Complexities

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |
| Min/Max | O(log n) | O(n) |

**Note**: Worst case happens with unbalanced tree (becomes linked list)

---

### Basic BST Operations

#### 1. Search in BST

```swift
func searchBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
    guard let root = root else { return nil }
    
    if val == root.val {
        return root
    } else if val < root.val {
        return searchBST(root.left, val)
    } else {
        return searchBST(root.right, val)
    }
}

// Iterative version (more efficient)
func searchBSTIterative(_ root: TreeNode?, _ val: Int) -> TreeNode? {
    var current = root
    
    while let node = current {
        if val == node.val {
            return node
        } else if val < node.val {
            current = node.left
        } else {
            current = node.right
        }
    }
    
    return nil
}
```

#### 2. Insert into BST

```swift
func insertIntoBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
    guard let root = root else {
        return TreeNode(val)
    }
    
    if val < root.val {
        root.left = insertIntoBST(root.left, val)
    } else {
        root.right = insertIntoBST(root.right, val)
    }
    
    return root
}

// Iterative version
func insertIntoBSTIterative(_ root: TreeNode?, _ val: Int) -> TreeNode? {
    let newNode = TreeNode(val)
    guard let root = root else { return newNode }
    
    var current = root
    
    while true {
        if val < current.val {
            if current.left == nil {
                current.left = newNode
                break
            }
            current = current.left!
        } else {
            if current.right == nil {
                current.right = newNode
                break
            }
            current = current.right!
        }
    }
    
    return root
}
```

#### 3. Delete from BST

```swift
// Most complex operation - 3 cases to handle
func deleteNode(_ root: TreeNode?, _ key: Int) -> TreeNode? {
    guard let root = root else { return nil }
    
    if key < root.val {
        root.left = deleteNode(root.left, key)
    } else if key > root.val {
        root.right = deleteNode(root.right, key)
    } else {
        // Found node to delete
        
        // Case 1: Leaf node or only one child
        if root.left == nil {
            return root.right
        }
        if root.right == nil {
            return root.left
        }
        
        // Case 2: Two children
        // Find inorder successor (smallest in right subtree)
        var successor = root.right
        while successor?.left != nil {
            successor = successor?.left
        }
        
        // Replace value with successor
        root.val = successor!.val
        
        // Delete successor
        root.right = deleteNode(root.right, successor!.val)
    }
    
    return root
}

// Helper: Find minimum node
func findMin(_ root: TreeNode?) -> TreeNode? {
    var current = root
    while current?.left != nil {
        current = current?.left
    }
    return current
}
```

#### 4. Find Min and Max

```swift
// Minimum: leftmost node
func findMin(_ root: TreeNode?) -> Int? {
    var current = root
    while current?.left != nil {
        current = current?.left
    }
    return current?.val
}

// Maximum: rightmost node
func findMax(_ root: TreeNode?) -> Int? {
    var current = root
    while current?.right != nil {
        current = current?.right
    }
    return current?.val
}
```

---

### Validate BST

```swift
// Check if tree is valid BST
func isValidBST(_ root: TreeNode?) -> Bool {
    return validate(root, nil, nil)
    
    func validate(_ node: TreeNode?, _ min: Int?, _ max: Int?) -> Bool {
        guard let node = node else { return true }
        
        // Check bounds
        if let min = min, node.val <= min {
            return false
        }
        if let max = max, node.val >= max {
            return false
        }
        
        // Recursively validate subtrees
        return validate(node.left, min, node.val) &&
               validate(node.right, node.val, max)
    }
}

// Using inorder traversal (should be sorted)
func isValidBST2(_ root: TreeNode?) -> Bool {
    var prev: Int? = nil
    
    func inorder(_ node: TreeNode?) -> Bool {
        guard let node = node else { return true }
        
        if !inorder(node.left) {
            return false
        }
        
        if let prevVal = prev, node.val <= prevVal {
            return false
        }
        prev = node.val
        
        return inorder(node.right)
    }
    
    return inorder(root)
}
```

---

### Classic BST Problems

#### Problem 1: Kth Smallest Element in BST

```swift
// Find kth smallest element (1-indexed)
func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
    var count = 0
    var result = 0
    
    func inorder(_ node: TreeNode?) {
        guard let node = node else { return }
        
        inorder(node.left)
        
        count += 1
        if count == k {
            result = node.val
            return
        }
        
        inorder(node.right)
    }
    
    inorder(root)
    return result
}

// Iterative version
func kthSmallestIterative(_ root: TreeNode?, _ k: Int) -> Int {
    var stack = [TreeNode]()
    var current = root
    var count = 0
    
    while current != nil || !stack.isEmpty {
        while current != nil {
            stack.append(current!)
            current = current?.left
        }
        
        current = stack.removeLast()
        count += 1
        
        if count == k {
            return current!.val
        }
        
        current = current?.right
    }
    
    return -1
}
```

#### Problem 2: Lowest Common Ancestor in BST

```swift
// Take advantage of BST property!
func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
    guard let root = root, let p = p, let q = q else { return nil }
    
    // Both in left subtree
    if p.val < root.val && q.val < root.val {
        return lowestCommonAncestor(root.left, p, q)
    }
    
    // Both in right subtree
    if p.val > root.val && q.val > root.val {
        return lowestCommonAncestor(root.right, p, q)
    }
    
    // Split point - this is the LCA
    return root
}

// Iterative
func lowestCommonAncestorIterative(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
    var current = root
    
    while let node = current {
        if p!.val < node.val && q!.val < node.val {
            current = node.left
        } else if p!.val > node.val && q!.val > node.val {
            current = node.right
        } else {
            return node
        }
    }
    
    return nil
}
```

#### Problem 3: Convert Sorted Array to BST

```swift
// Create balanced BST from sorted array
// [-10, -3, 0, 5, 9] → 
//      0
//     / \
//   -3   9
//   /   /
// -10  5

func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
    return buildBST(nums, 0, nums.count - 1)
    
    func buildBST(_ nums: [Int], _ left: Int, _ right: Int) -> TreeNode? {
        guard left <= right else { return nil }
        
        let mid = left + (right - left) / 2
        let root = TreeNode(nums[mid])
        
        root.left = buildBST(nums, left, mid - 1)
        root.right = buildBST(nums, mid + 1, right)
        
        return root
    }
}
```

#### Problem 4: Convert BST to Greater Tree

```swift
// Transform BST where each node's new value = 
// original value + sum of all greater values
// [4, 1, 6, 0, 2, 5, 7, null, null, null, 3, null, null, null, 8]
// Each node += sum of all nodes with greater values

func convertBST(_ root: TreeNode?) -> TreeNode? {
    var sum = 0
    
    func reverseInorder(_ node: TreeNode?) {
        guard let node = node else { return }
        
        // Visit right subtree first (larger values)
        reverseInorder(node.right)
        
        // Update current node
        sum += node.val
        node.val = sum
        
        // Visit left subtree
        reverseInorder(node.left)
    }
    
    reverseInorder(root)
    return root
}
```

#### Problem 5: Range Sum of BST

```swift
// Sum all values in range [low, high]
func rangeSumBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> Int {
    guard let root = root else { return 0 }
    
    var sum = 0
    
    // Include current node if in range
    if root.val >= low && root.val <= high {
        sum += root.val
    }
    
    // Only search left if there might be values >= low
    if root.val > low {
        sum += rangeSumBST(root.left, low, high)
    }
    
    // Only search right if there might be values <= high
    if root.val < high {
        sum += rangeSumBST(root.right, low, high)
    }
    
    return sum
}
```

#### Problem 6: Two Sum IV - Input is BST

```swift
// Find if there exist two elements that sum to k
func findTarget(_ root: TreeNode?, _ k: Int) -> Bool {
    var seen = Set<Int>()
    
    func search(_ node: TreeNode?) -> Bool {
        guard let node = node else { return false }
        
        let complement = k - node.val
        if seen.contains(complement) {
            return true
        }
        
        seen.insert(node.val)
        
        return search(node.left) || search(node.right)
    }
    
    return search(root)
}

// Using inorder + two pointers
func findTarget2(_ root: TreeNode?, _ k: Int) -> Bool {
    var values = [Int]()
    
    func inorder(_ node: TreeNode?) {
        guard let node = node else { return }
        inorder(node.left)
        values.append(node.val)
        inorder(node.right)
    }
    
    inorder(root)
    
    var left = 0
    var right = values.count - 1
    
    while left < right {
        let sum = values[left] + values[right]
        if sum == k {
            return true
        } else if sum < k {
            left += 1
        } else {
            right -= 1
        }
    }
    
    return false
}
```

#### Problem 7: Trim a BST

```swift
// Remove nodes outside range [low, high]
func trimBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> TreeNode? {
    guard let root = root else { return nil }
    
    // Current node is too small - trim left, return right subtree
    if root.val < low {
        return trimBST(root.right, low, high)
    }
    
    // Current node is too large - trim right, return left subtree
    if root.val > high {
        return trimBST(root.left, low, high)
    }
    
    // Current node is in range - trim both subtrees
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    
    return root
}
```

#### Problem 8: Inorder Successor in BST

```swift
// Find inorder successor of node (next larger value)
func inorderSuccessor(_ root: TreeNode?, _ p: TreeNode?) -> TreeNode? {
    guard let root = root, let p = p else { return nil }
    
    var successor: TreeNode? = nil
    var current = root
    
    while current != nil {
        if p.val < current!.val {
            successor = current
            current = current?.left
        } else {
            current = current?.right
        }
    }
    
    return successor
}

// If node has right child, successor is leftmost of right subtree
func inorderSuccessor2(_ node: TreeNode?) -> TreeNode? {
    guard let node = node else { return nil }
    
    if let right = node.right {
        var current = right
        while current.left != nil {
            current = current.left!
        }
        return current
    }
    
    return nil  // Need parent pointer for complete solution
}
```

#### Problem 9: Balance a BST

```swift
// Convert unbalanced BST to balanced BST
func balanceBST(_ root: TreeNode?) -> TreeNode? {
    // Step 1: Get sorted values via inorder
    var values = [Int]()
    
    func inorder(_ node: TreeNode?) {
        guard let node = node else { return }
        inorder(node.left)
        values.append(node.val)
        inorder(node.right)
    }
    
    inorder(root)
    
    // Step 2: Build balanced BST from sorted array
    func buildBalanced(_ left: Int, _ right: Int) -> TreeNode? {
        guard left <= right else { return nil }
        
        let mid = left + (right - left) / 2
        let node = TreeNode(values[mid])
        
        node.left = buildBalanced(left, mid - 1)
        node.right = buildBalanced(mid + 1, right)
        
        return node
    }
    
    return buildBalanced(0, values.count - 1)
}
```

#### Problem 10: Recover BST

```swift
// Two nodes are swapped, fix the BST
// Inorder should be sorted, find two violations

func recoverTree(_ root: TreeNode?) {
    var first: TreeNode? = nil
    var second: TreeNode? = nil
    var prev: TreeNode? = nil
    
    func inorder(_ node: TreeNode?) {
        guard let node = node else { return }
        
        inorder(node.left)
        
        // Found violation
        if let prevNode = prev, prevNode.val > node.val {
            if first == nil {
                first = prevNode
            }
            second = node
        }
        
        prev = node
        
        inorder(node.right)
    }
    
    inorder(root)
    
    // Swap values
    if let first = first, let second = second {
        let temp = first.val
        first.val = second.val
        second.val = temp
    }
}
```

---

### BST Iterator

```swift
// Implement iterator for BST (inorder traversal)
class BSTIterator {
    private var stack: [TreeNode] = []
    
    init(_ root: TreeNode?) {
        pushLeft(root)
    }
    
    func next() -> Int {
        let node = stack.removeLast()
        pushLeft(node.right)
        return node.val
    }
    
    func hasNext() -> Bool {
        return !stack.isEmpty
    }
    
    private func pushLeft(_ node: TreeNode?) {
        var current = node
        while current != nil {
            stack.append(current!)
            current = current?.left
        }
    }
}

// Usage
let bst = TreeNode(7)
bst.left = TreeNode(3)
bst.right = TreeNode(15)
bst.right?.left = TreeNode(9)
bst.right?.right = TreeNode(20)

let iterator = BSTIterator(bst)
print(iterator.next())     // 3
print(iterator.next())     // 7
print(iterator.hasNext())  // true
print(iterator.next())     // 9
```

---

### BST vs Hash Table

| Feature | BST | Hash Table |
|---------|-----|------------|
| Search | O(log n) | O(1) |
| Ordered | ✅ Yes | ❌ No |
| Range Query | ✅ Easy | ❌ Hard |
| Min/Max | O(log n) | O(n) |
| Space | O(n) | O(n) |

### When to Use BST

✅ **Use BST when:**
- Need sorted order
- Need range queries
- Need predecessor/successor
- Need to maintain order while inserting
- Space efficiency important (no hash overhead)

❌ **Don't use BST when:**
- Only need O(1) lookup (use hash table)
- Frequent insertions/deletions (tree can become unbalanced)
- Don't need ordering

### BST Tricks & Tips

1. **Inorder = Sorted**: Inorder traversal gives sorted sequence
2. **Bounds checking**: Pass min/max bounds for validation
3. **Morris Traversal**: O(1) space inorder traversal
4. **Predecessor/Successor**: Use BST property to find efficiently
5. **Convert to array**: Sometimes easier to work with sorted array
6. **Iterative > Recursive**: For space efficiency
7. **Parent pointers**: Helpful for predecessor/successor

### Common Patterns

| Pattern | Technique | Example |
|---------|-----------|---------|
| **Validation** | Pass bounds | Is valid BST |
| **Kth element** | Inorder with counter | Kth smallest |
| **Range queries** | Prune branches | Range sum |
| **Construction** | Binary search on array | Sorted array to BST |
| **Modification** | Reverse inorder | Greater tree |
| **Two pointers** | Inorder to array | Two sum |

---

<a name="chapter-18"></a>
## Chapter 18: Tree Traversals (Advanced)

Beyond basic traversals, there are many advanced tree traversal techniques and variations!

### Morris Traversal (O(1) Space)

Inorder traversal without recursion or stack!

```swift
// Morris Inorder Traversal - O(1) space
func morrisInorder(_ root: TreeNode?) -> [Int] {
    var result = [Int]()
    var current = root
    
    while current != nil {
        if current?.left == nil {
            // No left child, process current and go right
            result.append(current!.val)
            current = current?.right
        } else {
            // Find inorder predecessor
            var predecessor = current?.left
            while predecessor?.right != nil && predecessor?.right !== current {
                predecessor = predecessor?.right
            }
            
            if predecessor?.right == nil {
                // Create thread
                predecessor?.right = current
                current = current?.left
            } else {
                // Thread exists, remove it
                predecessor?.right = nil
                result.append(current!.val)
                current = current?.right
            }
        }
    }
    
    return result
}

// Morris Preorder
func morrisPreorder(_ root: TreeNode?) -> [Int] {
    var result = [Int]()
    var current = root
    
    while current != nil {
        if current?.left == nil {
            result.append(current!.val)
            current = current?.right
        } else {
            var predecessor = current?.left
            while predecessor?.right != nil && predecessor?.right !== current {
                predecessor = predecessor?.right
            }
            
            if predecessor?.right == nil {
                result.append(current!.val)  // Process before going left
                predecessor?.right = current
                current = current?.left
            } else {
                predecessor?.right = nil
                current = current?.right
            }
        }
    }
    
    return result
}
```

---

### Vertical Order Traversal

```swift
// Traverse tree vertically (column by column)
//       1
//      /  \
//     2    3
//    / \  / \
//   4  5 6  7
// Vertical: [[4], [2], [1, 5, 6], [3], [7]]

func verticalTraversal(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }
    
    var columnTable = [Int: [(row: Int, val: Int)]]()
    var queue = [(node: root, row: 0, col: 0)]
    
    while !queue.isEmpty {
        let (node, row, col) = queue.removeFirst()
        
        columnTable[col, default: []].append((row, node.val))
        
        if let left = node.left {
            queue.append((left, row + 1, col - 1))
        }
        if let right = node.right {
            queue.append((right, row + 1, col + 1))
        }
    }
    
    // Sort columns and values
    let sortedCols = columnTable.keys.sorted()
    var result = [[Int]]()
    
    for col in sortedCols {
        let sorted = columnTable[col]!.sorted { 
            $0.row == $1.row ? $0.val < $1.val : $0.row < $1.row 
        }
        result.append(sorted.map { $0.val })
    }
    
    return result
}
```

---

### Boundary Traversal

```swift
// Print boundary of tree (anti-clockwise)
// Left boundary → Leaves → Right boundary (reversed)

func boundaryOfBinaryTree(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [root.val]
    
    // Left boundary (excluding leaf)
    func addLeftBoundary(_ node: TreeNode?) {
        var current = node
        while current != nil {
            if !isLeaf(current) {
                result.append(current!.val)
            }
            current = current?.left ?? current?.right
        }
    }
    
    // Leaves
    func addLeaves(_ node: TreeNode?) {
        guard let node = node else { return }
        
        if isLeaf(node) {
            result.append(node.val)
            return
        }
        
        addLeaves(node.left)
        addLeaves(node.right)
    }
    
    // Right boundary (reversed, excluding leaf)
    func addRightBoundary(_ node: TreeNode?) {
        var current = node
        var temp = [Int]()
        
        while current != nil {
            if !isLeaf(current) {
                temp.append(current!.val)
            }
            current = current?.right ?? current?.left
        }
        
        result.append(contentsOf: temp.reversed())
    }
    
    func isLeaf(_ node: TreeNode?) -> Bool {
        return node?.left == nil && node?.right == nil
    }
    
    if !isLeaf(root) {
        addLeftBoundary(root.left)
        addLeaves(root)
        addRightBoundary(root.right)
    }
    
    return result
}
```

---

### Diagonal Traversal

```swift
// Traverse diagonally (right is same diagonal, left is next)
//       1
//      / \
//     2   3
//    / \
//   4   5
// Diagonals: [[1, 3], [2, 5], [4]]

func diagonalTraversal(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }
    
    var result = [[Int]]()
    var queue = [(node: root, diagonal: 0)]
    var diagonalMap = [Int: [Int]]()
    
    while !queue.isEmpty {
        let (node, diagonal) = queue.removeFirst()
        diagonalMap[diagonal, default: []].append(node.val)
        
        if let left = node.left {
            queue.append((left, diagonal + 1))
        }
        if let right = node.right {
            queue.append((right, diagonal))
        }
    }
    
    let maxDiagonal = diagonalMap.keys.max() ?? 0
    for i in 0...maxDiagonal {
        result.append(diagonalMap[i] ?? [])
    }
    
    return result
}
```

---

### Zigzag Level Order Traversal

```swift
// Level order but alternate directions
//     3
//    / \
//   9  20
//     /  \
//    15   7
// Result: [[3], [20, 9], [15, 7]]

func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }
    
    var result = [[Int]]()
    var queue = [root]
    var leftToRight = true
    
    while !queue.isEmpty {
        let levelSize = queue.count
        var currentLevel = [Int]()
        
        for _ in 0..<levelSize {
            let node = queue.removeFirst()
            currentLevel.append(node.val)
            
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
        
        if !leftToRight {
            currentLevel.reverse()
        }
        
        result.append(currentLevel)
        leftToRight.toggle()
    }
    
    return result
}
```

---

### Right Side View

```swift
// View tree from right side (rightmost node at each level)
//    1
//   / \
//  2   3
//   \   \
//    5   4
// Result: [1, 3, 4]

func rightSideView(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var result = [Int]()
    var queue = [root]
    
    while !queue.isEmpty {
        let levelSize = queue.count
        
        for i in 0..<levelSize {
            let node = queue.removeFirst()
            
            // Last node in level
            if i == levelSize - 1 {
                result.append(node.val)
            }
            
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
    }
    
    return result
}

// DFS approach
func rightSideViewDFS(_ root: TreeNode?) -> [Int] {
    var result = [Int]()
    
    func dfs(_ node: TreeNode?, _ level: Int) {
        guard let node = node else { return }
        
        // First time seeing this level
        if level == result.count {
            result.append(node.val)
        }
        
        // Visit right first
        dfs(node.right, level + 1)
        dfs(node.left, level + 1)
    }
    
    dfs(root, 0)
    return result
}
```

---

### Bottom View

```swift
// View tree from bottom (bottom-most node in each column)
func bottomView(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var columnMap = [Int: (row: Int, val: Int)]()
    var queue = [(node: root, row: 0, col: 0)]
    
    while !queue.isEmpty {
        let (node, row, col) = queue.removeFirst()
        
        // Update if this is lower row
        if columnMap[col] == nil || row >= columnMap[col]!.row {
            columnMap[col] = (row, node.val)
        }
        
        if let left = node.left {
            queue.append((left, row + 1, col - 1))
        }
        if let right = node.right {
            queue.append((right, row + 1, col + 1))
        }
    }
    
    let sortedCols = columnMap.keys.sorted()
    return sortedCols.map { columnMap[$0]!.val }
}
```

---

### Top View

```swift
// View tree from top (top-most node in each column)
func topView(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    
    var columnMap = [Int: Int]()
    var queue = [(node: root, col: 0)]
    
    while !queue.isEmpty {
        let (node, col) = queue.removeFirst()
        
        // Only set if not seen this column
        if columnMap[col] == nil {
            columnMap[col] = node.val
        }
        
        if let left = node.left {
            queue.append((left, col - 1))
        }
        if let right = node.right {
            queue.append((right, col + 1))
        }
    }
    
    let sortedCols = columnMap.keys.sorted()
    return sortedCols.map { columnMap[$0]! }
}
```

---

### All Paths from Root to Leaves

```swift
func allPaths(_ root: TreeNode?) -> [[Int]] {
    var result = [[Int]]()
    var path = [Int]()
    
    func dfs(_ node: TreeNode?) {
        guard let node = node else { return }
        
        path.append(node.val)
        
        // Leaf node
        if node.left == nil && node.right == nil {
            result.append(path)
        }
        
        dfs(node.left)
        dfs(node.right)
        
        path.removeLast()  // Backtrack
    }
    
    dfs(root)
    return result
}
```

---

### Sum of Distances in Tree

```swift
// For each node, sum of distances to all other nodes
func sumOfDistancesInTree(_ n: Int, _ edges: [[Int]]) -> [Int] {
    var graph = Array(repeating: [Int](), count: n)
    var count = Array(repeating: 1, count: n)  // Subtree size
    var result = Array(repeating: 0, count: n)
    
    // Build adjacency list
    for edge in edges {
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    }
    
    // Post-order: count subtree sizes and initial sums
    func postOrder(_ node: Int, _ parent: Int) {
        for neighbor in graph[node] {
            if neighbor == parent { continue }
            
            postOrder(neighbor, node)
            count[node] += count[neighbor]
            result[node] += result[neighbor] + count[neighbor]
        }
    }
    
    // Pre-order: adjust sums for each node
    func preOrder(_ node: Int, _ parent: Int) {
        for neighbor in graph[node] {
            if neighbor == parent { continue }
            
            result[neighbor] = result[node] - count[neighbor] + (n - count[neighbor])
            preOrder(neighbor, node)
        }
    }
    
    postOrder(0, -1)
    preOrder(0, -1)
    
    return result
}
```

---

### Traversal Comparison Table

| Traversal | Order | Use Case | Space |
|-----------|-------|----------|-------|
| **Inorder** | L-Root-R | BST sorted | O(h) |
| **Preorder** | Root-L-R | Copy tree | O(h) |
| **Postorder** | L-R-Root | Delete tree | O(h) |
| **Level Order** | Level by level | BFS, level problems | O(w) |
| **Morris** | L-Root-R | Space optimized | O(1) |
| **Vertical** | Column by column | Vertical views | O(n) |
| **Zigzag** | Alternate directions | Specific problems | O(w) |
| **Boundary** | Anti-clockwise | Tree outline | O(h) |

**h** = height, **w** = max width

---

### Advanced Traversal Patterns

1. **Multi-pass**: First pass collect info, second pass use it
2. **Level tracking**: Track current level/depth
3. **Column tracking**: Use horizontal distance
4. **Direction tracking**: Alternate or specific direction
5. **Custom ordering**: Sort by row, column, or value
6. **Parent tracking**: Keep parent reference for upward traversal
7. **Boundary detection**: Identify edges of tree

### Tips & Tricks

1. **BFS for level problems**: Use queue with level tracking
2. **DFS for path problems**: Use recursion with backtracking
3. **Morris for O(1) space**: Threading technique
4. **Hash map for column/level**: Group by position
5. **Deque for zigzag**: Add to front or back based on direction
6. **Multiple passes**: Sometimes easier than single pass
7. **Sentinel nodes**: Simplify boundary conditions

---

**🎯 Practice Problems:**
1. Maximum Width of Binary Tree
2. Binary Tree Cameras
3. Distribute Coins in Binary Tree
4. All Nodes Distance K in Binary Tree
5. Vertical Order Traversal of Binary Tree

<a name="chapter-19"></a>
## Chapter 19: Advanced Tree Problems

These problems combine multiple concepts and require deeper understanding of tree algorithms!

---

### Problem 1: Serialize and Deserialize Binary Tree

```swift
// Convert tree to string and back
class Codec {
    func serialize(_ root: TreeNode?) -> String {
        guard let root = root else { return "#" }
        
        return "\(root.val),\(serialize(root.left)),\(serialize(root.right))"
    }
    
    func deserialize(_ data: String) -> TreeNode? {
        var values = data.split(separator: ",").map(String.init)
        var index = 0
        
        return deserializeHelper(&values, &index)
    }
    
    private func deserializeHelper(_ values: inout [String], _ index: inout Int) -> TreeNode? {
        guard index < values.count else { return nil }
        
        let val = values[index]
        index += 1
        
        if val == "#" {
            return nil
        }
        
        let node = TreeNode(Int(val)!)
        node.left = deserializeHelper(&values, &index)
        node.right = deserializeHelper(&values, &index)
        
        return node
    }
}

// Level-order serialization
class CodecBFS {
    func serialize(_ root: TreeNode?) -> String {
        guard let root = root else { return "" }
        
        var result = [String]()
        var queue = [root]
        
        while !queue.isEmpty {
            let node = queue.removeFirst()
            result.append("\(node.val)")
            
            if let left = node.left {
                queue.append(left)
            } else {
                result.append("#")
            }
            
            if let right = node.right {
                queue.append(right)
            } else {
                result.append("#")
            }
        }
        
        return result.joined(separator: ",")
    }
    
    func deserialize(_ data: String) -> TreeNode? {
        guard !data.isEmpty else { return nil }
        
        let values = data.split(separator: ",").map(String.init)
        guard let rootVal = Int(values[0]) else { return nil }
        
        let root = TreeNode(rootVal)
        var queue = [root]
        var i = 1
        
        while !queue.isEmpty && i < values.count {
            let node = queue.removeFirst()
            
            if values[i] != "#" {
                node.left = TreeNode(Int(values[i])!)
                queue.append(node.left!)
            }
            i += 1
            
            if i < values.count && values[i] != "#" {
                node.right = TreeNode(Int(values[i])!)
                queue.append(node.right!)
            }
            i += 1
        }
        
        return root
    }
}
```

---

### Problem 2: Construct Binary Tree from Preorder and Inorder

```swift
// Preorder: [3, 9, 20, 15, 7]
// Inorder:  [9, 3, 15, 20, 7]
// Construct the tree

func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
    var preIndex = 0
    var inorderMap = [Int: Int]()
    
    // Map inorder values to indices
    for (i, val) in inorder.enumerated() {
        inorderMap[val] = i
    }
    
    return build(0, inorder.count - 1)
    
    func build(_ inStart: Int, _ inEnd: Int) -> TreeNode? {
        guard inStart <= inEnd else { return nil }
        
        let rootVal = preorder[preIndex]
        preIndex += 1
        
        let root = TreeNode(rootVal)
        let inIndex = inorderMap[rootVal]!
        
        // Build left first (preorder goes left before right)
        root.left = build(inStart, inIndex - 1)
        root.right = build(inIndex + 1, inEnd)
        
        return root
    }
}
```

---

### Problem 3: Construct Binary Tree from Inorder and Postorder

```swift
// Inorder:  [9, 3, 15, 20, 7]
// Postorder: [9, 15, 7, 20, 3]

func buildTreeFromInPost(_ inorder: [Int], _ postorder: [Int]) -> TreeNode? {
    var postIndex = postorder.count - 1
    var inorderMap = [Int: Int]()
    
    for (i, val) in inorder.enumerated() {
        inorderMap[val] = i
    }
    
    return build(0, inorder.count - 1)
    
    func build(_ inStart: Int, _ inEnd: Int) -> TreeNode? {
        guard inStart <= inEnd else { return nil }
        
        let rootVal = postorder[postIndex]
        postIndex -= 1
        
        let root = TreeNode(rootVal)
        let inIndex = inorderMap[rootVal]!
        
        // Build right first (postorder goes right before left when reading backwards)
        root.right = build(inIndex + 1, inEnd)
        root.left = build(inStart, inIndex - 1)
        
        return root
    }
}
```

---

### Problem 4: Maximum Path Sum

```swift
// Find maximum path sum (path can start and end anywhere)
//     -10
//     / \
//    9  20
//      /  \
//     15   7
// Maximum path: 15 -> 20 -> 7 = 42

func maxPathSum(_ root: TreeNode?) -> Int {
    var maxSum = Int.min
    
    func maxGain(_ node: TreeNode?) -> Int {
        guard let node = node else { return 0 }
        
        // Only take positive gains
        let leftGain = max(maxGain(node.left), 0)
        let rightGain = max(maxGain(node.right), 0)
        
        // Path through current node
        let pathSum = node.val + leftGain + rightGain
        maxSum = max(maxSum, pathSum)
        
        // Return max gain if continuing path
        return node.val + max(leftGain, rightGain)
    }
    
    maxGain(root)
    return maxSum
}
```

---

### Problem 5: All Nodes Distance K in Binary Tree

```swift
// Find all nodes at distance K from target node
func distanceK(_ root: TreeNode?, _ target: TreeNode?, _ k: Int) -> [Int] {
    var parentMap = [TreeNode: TreeNode]()
    
    // Build parent map
    func buildParentMap(_ node: TreeNode?, _ parent: TreeNode?) {
        guard let node = node else { return }
        if let parent = parent {
            parentMap[node] = parent
        }
        buildParentMap(node.left, node)
        buildParentMap(node.right, node)
    }
    
    buildParentMap(root, nil)
    
    // BFS from target
    guard let target = target else { return [] }
    
    var visited = Set<TreeNode>()
    var queue = [(node: target, distance: 0)]
    var result = [Int]()
    
    while !queue.isEmpty {
        let (node, distance) = queue.removeFirst()
        
        if visited.contains(node) {
            continue
        }
        visited.insert(node)
        
        if distance == k {
            result.append(node.val)
            continue
        }
        
        // Explore neighbors (left, right, parent)
        if let left = node.left {
            queue.append((left, distance + 1))
        }
        if let right = node.right {
            queue.append((right, distance + 1))
        }
        if let parent = parentMap[node] {
            queue.append((parent, distance + 1))
        }
    }
    
    return result
}
```

---

### Problem 6: Binary Tree Cameras

```swift
// Minimum cameras to monitor all nodes
// Camera monitors node, parent, and children
// 0 = not monitored, 1 = has camera, 2 = monitored

func minCameraCover(_ root: TreeNode?) -> Int {
    var cameras = 0
    
    func dfs(_ node: TreeNode?) -> Int {
        guard let node = node else { return 2 }  // Null is monitored
        
        let left = dfs(node.left)
        let right = dfs(node.right)
        
        // If any child not monitored, place camera here
        if left == 0 || right == 0 {
            cameras += 1
            return 1
        }
        
        // If any child has camera, this is monitored
        if left == 1 || right == 1 {
            return 2
        }
        
        // Both children monitored but no camera nearby
        return 0
    }
    
    // If root not monitored, add camera
    if dfs(root) == 0 {
        cameras += 1
    }
    
    return cameras
}
```

---

### Problem 7: Count Good Nodes in Binary Tree

```swift
// Node is "good" if no node in path from root has greater value
//       3
//      / \
//     1   4
//    /   / \
//   3   1   5
// Good nodes: 3, 3, 4, 5 = 4

func goodNodes(_ root: TreeNode?) -> Int {
    func dfs(_ node: TreeNode?, _ maxSoFar: Int) -> Int {
        guard let node = node else { return 0 }
        
        var count = 0
        
        if node.val >= maxSoFar {
            count = 1
        }
        
        let newMax = max(maxSoFar, node.val)
        count += dfs(node.left, newMax)
        count += dfs(node.right, newMax)
        
        return count
    }
    
    return dfs(root, Int.min)
}
```

---

### Problem 8: Distribute Coins in Binary Tree

```swift
// Each node has coins, need exactly 1 coin per node
// Find minimum moves (move coin to parent/child)
func distributeCoins(_ root: TreeNode?) -> Int {
    var moves = 0
    
    func dfs(_ node: TreeNode?) -> Int {
        guard let node = node else { return 0 }
        
        let leftExcess = dfs(node.left)
        let rightExcess = dfs(node.right)
        
        moves += abs(leftExcess) + abs(rightExcess)
        
        // Return excess/deficit for this subtree
        return node.val + leftExcess + rightExcess - 1
    }
    
    dfs(root)
    return moves
}
```

---

### Problem 9: Binary Tree Maximum Width

```swift
// Maximum width (max nodes in any level)
// Use indexing: left = 2*i, right = 2*i + 1
func widthOfBinaryTree(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    
    var maxWidth = 0
    var queue = [(node: root, index: 0)]
    
    while !queue.isEmpty {
        let levelSize = queue.count
        let leftmost = queue.first!.index
        let rightmost = queue.last!.index
        
        maxWidth = max(maxWidth, rightmost - leftmost + 1)
        
        for _ in 0..<levelSize {
            let (node, index) = queue.removeFirst()
            
            // Normalize indices to prevent overflow
            let normalizedIndex = index - leftmost
            
            if let left = node.left {
                queue.append((left, 2 * normalizedIndex))
            }
            if let right = node.right {
                queue.append((right, 2 * normalizedIndex + 1))
            }
        }
    }
    
    return maxWidth
}
```

---

### Problem 10: House Robber III

```swift
// Rob houses in binary tree (can't rob adjacent nodes)
// Return maximum money
func rob(_ root: TreeNode?) -> Int {
    let result = robHelper(root)
    return max(result.withRoot, result.withoutRoot)
    
    func robHelper(_ node: TreeNode?) -> (withRoot: Int, withoutRoot: Int) {
        guard let node = node else {
            return (0, 0)
        }
        
        let left = robHelper(node.left)
        let right = robHelper(node.right)
        
        // Rob this node: can't rob children
        let withRoot = node.val + left.withoutRoot + right.withoutRoot
        
        // Don't rob this node: take max from children
        let withoutRoot = max(left.withRoot, left.withoutRoot) + 
                         max(right.withRoot, right.withoutRoot)
        
        return (withRoot, withoutRoot)
    }
}
```

---

### Problem 11: Longest Univalue Path

```swift
// Longest path with same value
//       5
//      / \
//     4   5
//    / \   \
//   1   1   5
// Result: 2 (5->5->5)

func longestUnivaluePath(_ root: TreeNode?) -> Int {
    var maxLength = 0
    
    func dfs(_ node: TreeNode?) -> Int {
        guard let node = node else { return 0 }
        
        let left = dfs(node.left)
        let right = dfs(node.right)
        
        var leftPath = 0
        var rightPath = 0
        
        if let leftChild = node.left, leftChild.val == node.val {
            leftPath = left + 1
        }
        
        if let rightChild = node.right, rightChild.val == node.val {
            rightPath = right + 1
        }
        
        maxLength = max(maxLength, leftPath + rightPath)
        
        return max(leftPath, rightPath)
    }
    
    dfs(root)
    return maxLength
}
```

---

### Problem 12: Sum Root to Leaf Numbers

```swift
// Sum all root-to-leaf numbers
//    1
//   / \
//  2   3
// Numbers: 12, 13 → Sum = 25

func sumNumbers(_ root: TreeNode?) -> Int {
    func dfs(_ node: TreeNode?, _ currentSum: Int) -> Int {
        guard let node = node else { return 0 }
        
        let newSum = currentSum * 10 + node.val
        
        // Leaf node
        if node.left == nil && node.right == nil {
            return newSum
        }
        
        return dfs(node.left, newSum) + dfs(node.right, newSum)
    }
    
    return dfs(root, 0)
}
```

---

### Problem 13: Binary Tree Pruning

```swift
// Remove subtrees where all values are 0
func pruneTree(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    
    root.left = pruneTree(root.left)
    root.right = pruneTree(root.right)
    
    // If this is a leaf with value 0, remove it
    if root.val == 0 && root.left == nil && root.right == nil {
        return nil
    }
    
    return root
}
```

---

### Problem 14: Insufficient Nodes in Root to Leaf Paths

```swift
// Remove nodes where path sum < limit
func sufficientSubset(_ root: TreeNode?, _ limit: Int) -> TreeNode? {
    guard let root = root else { return nil }
    
    // Leaf node
    if root.left == nil && root.right == nil {
        return root.val < limit ? nil : root
    }
    
    // Recursively process subtrees
    root.left = sufficientSubset(root.left, limit - root.val)
    root.right = sufficientSubset(root.right, limit - root.val)
    
    // If both children removed, remove this node
    if root.left == nil && root.right == nil {
        return nil
    }
    
    return root
}
```

---

### Problem 15: Subtree of Another Tree

```swift
// Check if subRoot is subtree of root
func isSubtree(_ root: TreeNode?, _ subRoot: TreeNode?) -> Bool {
    guard let root = root else { return subRoot == nil }
    
    if isSameTree(root, subRoot) {
        return true
    }
    
    return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot)
    
    func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        if p == nil && q == nil { return true }
        if p == nil || q == nil { return false }
        
        return p!.val == q!.val &&
               isSameTree(p!.left, q!.left) &&
               isSameTree(p!.right, q!.right)
    }
}
```

---

### Advanced Tree Patterns

| Pattern | Description | Example Problems |
|---------|-------------|------------------|
| **State Tracking** | Track state up/down tree | Good nodes, path sum |
| **Bottom-Up** | Process children first | Max path sum, distribute coins |
| **Two-State DP** | Track with/without current | House robber III |
| **Parent Tracking** | Build parent map | Distance K |
| **Indexing** | Assign indices to nodes | Max width |
| **Pruning** | Remove based on condition | Binary tree pruning |
| **Construction** | Build from traversals | Construct from pre/in |
| **Serialization** | Convert to/from string | Serialize/deserialize |

---

### Problem-Solving Framework for Trees

```
1. Identify the pattern:
   - Single pass or multiple passes?
   - Top-down or bottom-up?
   - Need parent references?
   - Level-based or path-based?

2. Choose traversal:
   - DFS: Paths, height, most problems
   - BFS: Level-based, shortest path, width
   - Inorder: BST problems
   - Custom: Specific traversal needs

3. Track state:
   - Return values from recursion
   - Pass parameters down
   - Use global/class variables
   - Build auxiliary structures (maps, sets)

4. Handle edge cases:
   - Null nodes
   - Single node
   - Unbalanced trees
   - Negative values
```

---

### Tips for Hard Tree Problems

1. **Draw it out**: Visualize small examples
2. **Start simple**: Handle base cases first
3. **Think recursively**: What do subtrees return?
4. **Use helper functions**: Add parameters as needed
5. **Track multiple states**: Often need 2+ values
6. **Post-order for bottom-up**: Process children first
7. **Pre-order for top-down**: Pass state down
8. **BFS for level problems**: Queue-based traversal
9. **Parent map**: When need to go upwards
10. **Index tracking**: For width/position problems

---

<a name="chapter-20"></a>
## Chapter 20: Heap Implementation

A heap is a complete binary tree where parent is always greater (max heap) or smaller (min heap) than children. Perfect for priority queues!

### Heap Properties

```
Max Heap:          Min Heap:
    100               1
   /   \            /   \
  19   36          2     3
 / \   /          / \   /
17 3  25         17 19 36

Max Heap: Parent >= Children
Min Heap: Parent <= Children
```

### Array Representation

```swift
// Heap stored as array
// For index i:
// - Left child:  2*i + 1
// - Right child: 2*i + 2
// - Parent:      (i - 1) / 2

// Example: [100, 19, 36, 17, 3, 25]
//          Index: 0   1   2   3  4  5
//
//          100 (0)
//         /       \
//       19(1)     36(2)
//      /   \       /
//    17(3) 3(4) 25(5)
```

---

### Min Heap Implementation

```swift
struct MinHeap<T: Comparable> {
    private var elements: [T] = []
    
    var isEmpty: Bool {
        return elements.isEmpty
    }
    
    var count: Int {
        return elements.count
    }
    
    var peek: T? {
        return elements.first
    }
    
    // Insert element - O(log n)
    mutating func insert(_ element: T) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }
    
    // Remove minimum - O(log n)
    mutating func extractMin() -> T? {
        guard !elements.isEmpty else { return nil }
        
        if elements.count == 1 {
            return elements.removeLast()
        }
        
        let min = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        
        return min
    }
    
    // Build heap from array - O(n)
    init(_ array: [T]) {
        elements = array
        
        // Start from last parent node
        if !elements.isEmpty {
            for i in stride(from: elements.count / 2 - 1, through: 0, by: -1) {
                siftDown(from: i)
            }
        }
    }
    
    // Heapify up
    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        
        while child > 0 && elements[child] < elements[parent] {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }
    
    // Heapify down
    private mutating func siftDown(from index: Int) {
        var parent = index
        
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var smallest = parent
            
            if left < elements.count && elements[left] < elements[smallest] {
                smallest = left
            }
            
            if right < elements.count && elements[right] < elements[smallest] {
                smallest = right
            }
            
            if smallest == parent {
                break
            }
            
            elements.swapAt(parent, smallest)
            parent = smallest
        }
    }
}

// Usage
var minHeap = MinHeap<Int>()
minHeap.insert(5)
minHeap.insert(3)
minHeap.insert(7)
minHeap.insert(1)

print(minHeap.extractMin()!)  // 1
print(minHeap.extractMin()!)  // 3
print(minHeap.peek!)          // 5
```

---

### Max Heap Implementation

```swift
struct MaxHeap<T: Comparable> {
    private var elements: [T] = []
    
    var isEmpty: Bool {
        return elements.isEmpty
    }
    
    var count: Int {
        return elements.count
    }
    
    var peek: T? {
        return elements.first
    }
    
    mutating func insert(_ element: T) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }
    
    mutating func extractMax() -> T? {
        guard !elements.isEmpty else { return nil }
        
        if elements.count == 1 {
            return elements.removeLast()
        }
        
        let max = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        
        return max
    }
    
    init(_ array: [T]) {
        elements = array
        if !elements.isEmpty {
            for i in stride(from: elements.count / 2 - 1, through: 0, by: -1) {
                siftDown(from: i)
            }
        }
    }
    
    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        
        while child > 0 && elements[child] > elements[parent] {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }
    
    private mutating func siftDown(from index: Int) {
        var parent = index
        
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var largest = parent
            
            if left < elements.count && elements[left] > elements[largest] {
                largest = left
            }
            
            if right < elements.count && elements[right] > elements[largest] {
                largest = right
            }
            
            if largest == parent {
                break
            }
            
            elements.swapAt(parent, largest)
            parent = largest
        }
    }
}
```

---

### Generic Priority Queue

```swift
struct PriorityQueue<T: Comparable> {
    private var heap: [T] = []
    private let isMinHeap: Bool
    
    init(isMinHeap: Bool = true) {
        self.isMinHeap = isMinHeap
    }
    
    var isEmpty: Bool {
        return heap.isEmpty
    }
    
    var count: Int {
        return heap.count
    }
    
    var peek: T? {
        return heap.first
    }
    
    mutating func enqueue(_ element: T) {
        heap.append(element)
        siftUp(from: heap.count - 1)
    }
    
    mutating func dequeue() -> T? {
        guard !heap.isEmpty else { return nil }
        
        if heap.count == 1 {
            return heap.removeLast()
        }
        
        let element = heap[0]
        heap[0] = heap.removeLast()
        siftDown(from: 0)
        
        return element
    }
    
    private func higherPriority(_ i: Int, _ j: Int) -> Bool {
        return isMinHeap ? heap[i] < heap[j] : heap[i] > heap[j]
    }
    
    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        
        while child > 0 && higherPriority(child, parent) {
            heap.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }
    
    private mutating func siftDown(from index: Int) {
        var parent = index
        
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent
            
            if left < heap.count && higherPriority(left, candidate) {
                candidate = left
            }
            
            if right < heap.count && higherPriority(right, candidate) {
                candidate = right
            }
            
            if candidate == parent {
                break
            }
            
            heap.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

// Usage
var maxPQ = PriorityQueue<Int>(isMinHeap: false)
maxPQ.enqueue(5)
maxPQ.enqueue(10)
maxPQ.enqueue(3)
print(maxPQ.dequeue()!)  // 10
```

---

### Heap Operations Summary

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| peek() | O(1) | View top element |
| insert() | O(log n) | Add element |
| extract() | O(log n) | Remove top |
| build() | O(n) | Create from array |
| search() | O(n) | Find element |
| delete() | O(log n) | Remove specific |

---

### Classic Heap Problems

#### Problem 1: Kth Largest Element

```swift
// Find kth largest element in array
func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
    var minHeap = MinHeap<Int>()
    
    for num in nums {
        minHeap.insert(num)
        
        if minHeap.count > k {
            minHeap.extractMin()
        }
    }
    
    return minHeap.peek!
}

// Using Swift's built-in sorting
func findKthLargest2(_ nums: [Int], _ k: Int) -> Int {
    return nums.sorted(by: >)[k - 1]
}

print(findKthLargest([3, 2, 1, 5, 6, 4], 2))  // 5
```

#### Problem 2: Top K Frequent Elements

```swift
func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
    var freq = [Int: Int]()
    
    for num in nums {
        freq[num, default: 0] += 1
    }
    
    // Min heap of size k
    var heap = [(count: Int, num: Int)]()
    
    for (num, count) in freq {
        heap.append((count, num))
        heap.sort { $0.count < $1.count }
        
        if heap.count > k {
            heap.removeFirst()
        }
    }
    
    return heap.map { $0.num }
}
```

#### Problem 3: Merge K Sorted Lists

```swift
// Using heap
func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
    guard !lists.isEmpty else { return nil }
    
    var heap = [(val: Int, node: ListNode)]()
    
    // Add first node from each list
    for list in lists {
        if let node = list {
            heap.append((node.val, node))
        }
    }
    
    heap.sort { $0.val < $1.val }
    
    let dummy = ListNode(0)
    var current = dummy
    
    while !heap.isEmpty {
        let minPair = heap.removeFirst()
        let node = minPair.node
        
        current.next = node
        current = current.next!
        
        if let next = node.next {
            // Insert in sorted position
            var inserted = false
            for i in 0..<heap.count {
                if next.val < heap[i].val {
                    heap.insert((next.val, next), at: i)
                    inserted = true
                    break
                }
            }
            if !inserted {
                heap.append((next.val, next))
            }
        }
    }
    
    return dummy.next
}
```

#### Problem 4: Find Median from Data Stream

```swift
class MedianFinder {
    private var maxHeap: MaxHeap<Int> = MaxHeap([])  // Lower half
    private var minHeap: MinHeap<Int> = MinHeap([])  // Upper half
    
    func addNum(_ num: Int) {
        // Add to maxHeap first
        maxHeap.insert(num)
        
        // Balance: move max from maxHeap to minHeap
        if let maxVal = maxHeap.extractMax() {
            minHeap.insert(maxVal)
        }
        
        // Ensure maxHeap has more or equal elements
        if minHeap.count > maxHeap.count {
            if let minVal = minHeap.extractMin() {
                maxHeap.insert(minVal)
            }
        }
    }
    
    func findMedian() -> Double {
        if maxHeap.count > minHeap.count {
            return Double(maxHeap.peek!)
        } else {
            return Double(maxHeap.peek! + minHeap.peek!) / 2.0
        }
    }
}

let mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  // 1.5
mf.addNum(3)
print(mf.findMedian())  // 2.0
```

#### Problem 5: Kth Smallest Element in Sorted Matrix

```swift
// Matrix is sorted row and column wise
func kthSmallest(_ matrix: [[Int]], _ k: Int) -> Int {
    let n = matrix.count
    var heap = [(val: Int, row: Int, col: Int)]()
    
    // Add first element of each row
    for i in 0..<min(n, k) {
        heap.append((matrix[i][0], i, 0))
    }
    
    heap.sort { $0.val < $1.val }
    
    var count = 0
    var result = 0
    
    while !heap.isEmpty {
        let min = heap.removeFirst()
        result = min.val
        count += 1
        
        if count == k {
            break
        }
        
        // Add next element from same row
        if min.col + 1 < n {
            let next = (matrix[min.row][min.col + 1], min.row, min.col + 1)
            
            var inserted = false
            for i in 0..<heap.count {
                if next.0 < heap[i].val {
                    heap.insert(next, at: i)
                    inserted = true
                    break
                }
            }
            if !inserted {
                heap.append(next)
            }
        }
    }
    
    return result
}
```

---

### Heap Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Top K** | Find k largest/smallest | Kth largest element |
| **K-way merge** | Merge k sorted arrays | Merge k sorted lists |
| **Running median** | Two heaps | Median from stream |
| **Frequency** | Count + heap | Top k frequent |
| **Intervals** | Sort + heap | Meeting rooms |

### Tips & Tricks

1. **Min heap for K largest**: Keep k largest in min heap
2. **Max heap for K smallest**: Keep k smallest in max heap
3. **Two heaps for median**: Max heap (left), min heap (right)
4. **Heapify is O(n)**: Building heap from array
5. **Custom comparator**: For complex objects
6. **Index calculations**: Parent = (i-1)/2, Left = 2i+1, Right = 2i+2
7. **Complete tree**: Always fill left to right

---

**🎯 Practice Problems:**
1. Last Stone Weight
2. Reorganize String
3. Furthest Building You Can Reach
4. Find K Pairs with Smallest Sums
5. IPO (Maximum Capital)

<a name="chapter-21"></a>
## Chapter 21: Priority Queue Problems

Priority queues (heaps) are essential for problems involving ordering, scheduling, and optimization!

---

### Problem 1: Last Stone Weight

```swift
// Smash two heaviest stones, difference remains
// [2, 7, 4, 1, 8, 1] → 1

func lastStoneWeight(_ stones: [Int]) -> Int {
    var maxHeap = MaxHeap(stones)
    
    while maxHeap.count > 1 {
        let first = maxHeap.extractMax()!
        let second = maxHeap.extractMax()!
        
        if first != second {
            maxHeap.insert(first - second)
        }
    }
    
    return maxHeap.isEmpty ? 0 : maxHeap.peek!
}

print(lastStoneWeight([2, 7, 4, 1, 8, 1]))  // 1
```

---

### Problem 2: K Closest Points to Origin

```swift
// Find k closest points to origin (0, 0)
// [[1, 3], [-2, 2]], k = 1 → [[-2, 2]]

func kClosest(_ points: [[Int]], _ k: Int) -> [[Int]] {
    // Max heap to keep k closest (using negative distance for max heap behavior)
    var heap = [(dist: Int, point: [Int])]()
    
    for point in points {
        let distance = point[0] * point[0] + point[1] * point[1]
        heap.append((distance, point))
        heap.sort { $0.dist > $1.dist }  // Max heap
        
        if heap.count > k {
            heap.removeFirst()
        }
    }
    
    return heap.map { $0.point }
}

// More efficient with proper max heap
func kClosest2(_ points: [[Int]], _ k: Int) -> [[Int]] {
    let sorted = points.sorted { point1, point2 in
        let dist1 = point1[0] * point1[0] + point1[1] * point1[1]
        let dist2 = point2[0] * point2[0] + point2[1] * point2[1]
        return dist1 < dist2
    }
    
    return Array(sorted.prefix(k))
}

print(kClosest([[1, 3], [-2, 2]], 1))  // [[-2, 2]]
```

---

### Problem 3: Reorganize String

```swift
// Reorganize so no two adjacent characters are same
// "aab" → "aba"
// "aaab" → "" (impossible)

func reorganizeString(_ s: String) -> String {
    var freq = [Character: Int]()
    
    for char in s {
        freq[char, default: 0] += 1
    }
    
    // Max heap by frequency
    var heap = [(count: Int, char: Character)]()
    for (char, count) in freq {
        heap.append((count, char))
    }
    heap.sort { $0.count > $1.count }
    
    // Check if possible
    let maxFreq = heap.first!.count
    if maxFreq > (s.count + 1) / 2 {
        return ""
    }
    
    var result = [Character]()
    var prev: (count: Int, char: Character)? = nil
    
    while !heap.isEmpty {
        var current = heap.removeFirst()
        result.append(current.char)
        current.count -= 1
        
        // Add back previous if still has count
        if let prev = prev, prev.count > 0 {
            heap.append(prev)
            heap.sort { $0.count > $1.count }
        }
        
        prev = current.count > 0 ? current : nil
    }
    
    return result.count == s.count ? String(result) : ""
}

print(reorganizeString("aab"))    // "aba"
print(reorganizeString("aaab"))   // ""
```

---

### Problem 4: Task Scheduler

```swift
// Tasks with cooldown n between same tasks
// ["A","A","A","B","B","B"], n = 2 → 8
// A -> B -> idle -> A -> B -> idle -> A -> B

func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
    var freq = [Character: Int]()
    
    for task in tasks {
        freq[task, default: 0] += 1
    }
    
    let maxFreq = freq.values.max()!
    let maxCount = freq.values.filter { $0 == maxFreq }.count
    
    let partCount = maxFreq - 1
    let partLength = n - (maxCount - 1)
    let emptySlots = partCount * partLength
    let availableTasks = tasks.count - maxFreq * maxCount
    let idles = max(0, emptySlots - availableTasks)
    
    return tasks.count + idles
}

// Alternative: simulation with heap
func leastInterval2(_ tasks: [Character], _ n: Int) -> Int {
    var freq = [Character: Int]()
    
    for task in tasks {
        freq[task, default: 0] += 1
    }
    
    var heap = freq.values.sorted(by: >)
    var time = 0
    
    while !heap.isEmpty {
        var temp = [Int]()
        
        for i in 0...n {
            if !heap.isEmpty {
                let count = heap.removeFirst()
                if count - 1 > 0 {
                    temp.append(count - 1)
                }
            }
            
            time += 1
            
            if heap.isEmpty && temp.isEmpty {
                break
            }
        }
        
        heap.append(contentsOf: temp)
        heap.sort(by: >)
    }
    
    return time
}

print(leastInterval(["A","A","A","B","B","B"], 2))  // 8
```

---

### Problem 5: Furthest Building You Can Reach

```swift
// Use ladders/bricks to climb buildings
// Can use ladder for any height, bricks equal to height diff
func furthestBuilding(_ heights: [Int], _ bricks: Int, _ ladders: Int) -> Int {
    var minHeap = MinHeap<Int>()
    var bricksUsed = 0
    
    for i in 0..<heights.count - 1 {
        let diff = heights[i + 1] - heights[i]
        
        if diff <= 0 {
            continue  // Going down, no resources needed
        }
        
        minHeap.insert(diff)
        
        // If used more ladders than available, use bricks for smallest climb
        if minHeap.count > ladders {
            bricksUsed += minHeap.extractMin()!
        }
        
        // Not enough bricks
        if bricksUsed > bricks {
            return i
        }
    }
    
    return heights.count - 1
}

print(furthestBuilding([4, 2, 7, 6, 9, 14, 12], 5, 1))  // 4
```

---

### Problem 6: Find K Pairs with Smallest Sums

```swift
// Find k pairs with smallest sums from two sorted arrays
// nums1 = [1, 7, 11], nums2 = [2, 4, 6], k = 3
// → [[1, 2], [1, 4], [1, 6]]

func kSmallestPairs(_ nums1: [Int], _ nums2: [Int], _ k: Int) -> [[Int]] {
    guard !nums1.isEmpty && !nums2.isEmpty else { return [] }
    
    var result = [[Int]]()
    var heap = [(sum: Int, i: Int, j: Int)]()
    
    // Initialize with first element of nums1 paired with all of nums2
    for j in 0..<min(k, nums2.count) {
        heap.append((nums1[0] + nums2[j], 0, j))
    }
    
    heap.sort { $0.sum < $1.sum }
    
    while !heap.isEmpty && result.count < k {
        let (_, i, j) = heap.removeFirst()
        result.append([nums1[i], nums2[j]])
        
        // Add next pair from nums1
        if i + 1 < nums1.count {
            let newPair = (nums1[i + 1] + nums2[j], i + 1, j)
            
            var inserted = false
            for idx in 0..<heap.count {
                if newPair.sum < heap[idx].sum {
                    heap.insert(newPair, at: idx)
                    inserted = true
                    break
                }
            }
            if !inserted {
                heap.append(newPair)
            }
        }
    }
    
    return result
}

print(kSmallestPairs([1, 7, 11], [2, 4, 6], 3))
// [[1, 2], [1, 4], [1, 6]]
```

---

### Problem 7: Ugly Number II

```swift
// Find nth ugly number (only prime factors 2, 3, 5)
// First 10: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12

func nthUglyNumber(_ n: Int) -> Int {
    var ugly = [1]
    var i2 = 0, i3 = 0, i5 = 0
    
    while ugly.count < n {
        let next2 = ugly[i2] * 2
        let next3 = ugly[i3] * 3
        let next5 = ugly[i5] * 5
        
        let nextUgly = min(next2, next3, next5)
        ugly.append(nextUgly)
        
        if nextUgly == next2 { i2 += 1 }
        if nextUgly == next3 { i3 += 1 }
        if nextUgly == next5 { i5 += 1 }
    }
    
    return ugly[n - 1]
}

// Using heap
func nthUglyNumber2(_ n: Int) -> Int {
    var heap = MinHeap([1])
    var seen = Set([1])
    var ugly = 1
    
    for _ in 0..<n {
        ugly = heap.extractMin()!
        
        for factor in [2, 3, 5] {
            let newUgly = ugly * factor
            if !seen.contains(newUgly) {
                seen.insert(newUgly)
                heap.insert(newUgly)
            }
        }
    }
    
    return ugly
}

print(nthUglyNumber(10))  // 12
```

---

### Problem 8: Sliding Window Maximum (Using Deque)

```swift
// Already covered, but important heap application
func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    var result = [Int]()
    var deque = [Int]()  // Indices
    
    for i in 0..<nums.count {
        // Remove out of window
        while !deque.isEmpty && deque.first! <= i - k {
            deque.removeFirst()
        }
        
        // Remove smaller elements
        while !deque.isEmpty && nums[deque.last!] < nums[i] {
            deque.removeLast()
        }
        
        deque.append(i)
        
        if i >= k - 1 {
            result.append(nums[deque.first!])
        }
    }
    
    return result
}
```

---

### Problem 9: Meeting Rooms II

```swift
// Minimum meeting rooms needed
// [[0, 30], [5, 10], [15, 20]] → 2

func minMeetingRooms(_ intervals: [[Int]]) -> Int {
    guard !intervals.isEmpty else { return 0 }
    
    // Sort by start time
    let sorted = intervals.sorted { $0[0] < $1[0] }
    
    // Min heap of end times
    var heap = MinHeap([sorted[0][1]])
    
    for i in 1..<sorted.count {
        let start = sorted[i][0]
        let end = sorted[i][1]
        
        // If earliest ending meeting finished, reuse room
        if start >= heap.peek! {
            heap.extractMin()
        }
        
        heap.insert(end)
    }
    
    return heap.count
}

print(minMeetingRooms([[0, 30], [5, 10], [15, 20]]))  // 2
```

---

### Problem 10: IPO (Maximum Capital)

```swift
// Maximize capital with k projects
// Each project has profit and capital requirement
func findMaximizedCapital(_ k: Int, _ w: Int, _ profits: [Int], _ capital: [Int]) -> Int {
    let n = profits.count
    var projects = [(capital: Int, profit: Int)]()
    
    for i in 0..<n {
        projects.append((capital[i], profits[i]))
    }
    
    // Sort by capital requirement
    projects.sort { $0.capital < $1.capital }
    
    var currentCapital = w
    var maxProfitHeap = MaxHeap<Int>([])
    var projectIndex = 0
    
    for _ in 0..<k {
        // Add all affordable projects to heap
        while projectIndex < n && projects[projectIndex].capital <= currentCapital {
            maxProfitHeap.insert(projects[projectIndex].profit)
            projectIndex += 1
        }
        
        // If no project affordable, break
        guard let profit = maxProfitHeap.extractMax() else {
            break
        }
        
        currentCapital += profit
    }
    
    return currentCapital
}

print(findMaximizedCapital(2, 0, [1, 2, 3], [0, 1, 1]))  // 4
```

---

### Problem 11: Sort Characters by Frequency

```swift
// Sort characters by decreasing frequency
// "tree" → "eert" or "eetr"

func frequencySort(_ s: String) -> String {
    var freq = [Character: Int]()
    
    for char in s {
        freq[char, default: 0] += 1
    }
    
    // Max heap by frequency
    var heap = freq.map { (count: $0.value, char: $0.key) }
    heap.sort { $0.count > $1.count }
    
    var result = ""
    for (count, char) in heap {
        result += String(repeating: char, count: count)
    }
    
    return result
}

print(frequencySort("tree"))  // "eert"
```

---

### Problem 12: Super Ugly Number

```swift
// Ugly number with given primes
// n = 12, primes = [2, 7, 13, 19] → 32

func nthSuperUglyNumber(_ n: Int, _ primes: [Int]) -> Int {
    var ugly = [1]
    var indices = Array(repeating: 0, count: primes.count)
    
    while ugly.count < n {
        var nextUgly = Int.max
        
        // Find next smallest ugly number
        for i in 0..<primes.count {
            nextUgly = min(nextUgly, ugly[indices[i]] * primes[i])
        }
        
        ugly.append(nextUgly)
        
        // Move pointers that generated this ugly number
        for i in 0..<primes.count {
            if nextUgly == ugly[indices[i]] * primes[i] {
                indices[i] += 1
            }
        }
    }
    
    return ugly[n - 1]
}

print(nthSuperUglyNumber(12, [2, 7, 13, 19]))  // 32
```

---

### Priority Queue Patterns Summary

| Pattern | Description | Example |
|---------|-------------|---------|
| **K-th element** | Keep k elements in heap | Kth largest |
| **Merge K sorted** | Heap of k elements | Merge k lists |
| **Scheduling** | Order by time/priority | Task scheduler |
| **Running median** | Two heaps | Median finder |
| **Greedy selection** | Always pick best | IPO |
| **Frequency-based** | Order by count | Top k frequent |
| **Distance-based** | Order by distance | K closest points |

### When to Use Priority Queue

✅ **Use Priority Queue when:**
- Need repeated min/max operations
- K-th element problems
- Merging sorted sequences
- Scheduling/ordering tasks
- Running statistics (median, etc.)
- Greedy algorithms with selection

### Tips & Tricks

1. **Min heap for K largest**: Keep only k elements
2. **Max heap for K smallest**: Keep only k elements
3. **Two heaps for median**: Balance sizes
4. **Lazy deletion**: Mark as deleted, clean later
5. **Custom comparator**: For complex ordering
6. **Size constraint**: Limit heap size for space efficiency
7. **Combine with sorting**: Sometimes sort first, then heap

---

<a name="chapter-22"></a>
## Chapter 22: Graph Representations

Graphs are collections of vertices (nodes) connected by edges. They model networks, relationships, and many real-world problems!

### Graph Terminology

```
Vertices/Nodes: Points in graph (A, B, C)
Edges: Connections between vertices
Degree: Number of edges connected to vertex
Path: Sequence of vertices
Cycle: Path that starts and ends at same vertex
Connected: Path exists between all vertex pairs
```

### Types of Graphs

```swift
// 1. Undirected Graph
//    A --- B
//    |     |
//    C --- D
// Edge between A-B means both A→B and B→A

// 2. Directed Graph (Digraph)
//    A → B
//    ↓   ↓
//    C → D
// Edge A→B means only one direction

// 3. Weighted Graph
//    A --5-- B
//    |       |
//    3       2
//    |       |
//    C --4-- D
// Edges have weights/costs

// 4. Tree (special graph)
//    - Connected
//    - Acyclic
//    - n vertices, n-1 edges
```

---

### Graph Representation 1: Adjacency Matrix

```swift
// 2D array where matrix[i][j] = 1 if edge exists
class GraphMatrix {
    var matrix: [[Int]]
    let vertices: Int
    
    init(vertices: Int) {
        self.vertices = vertices
        self.matrix = Array(repeating: Array(repeating: 0, count: vertices), 
                           count: vertices)
    }
    
    // Add undirected edge
    func addEdge(_ u: Int, _ v: Int) {
        matrix[u][v] = 1
        matrix[v][u] = 1
    }
    
    // Add directed edge
    func addDirectedEdge(_ u: Int, _ v: Int) {
        matrix[u][v] = 1
    }
    
    // Add weighted edge
    func addWeightedEdge(_ u: Int, _ v: Int, _ weight: Int) {
        matrix[u][v] = weight
        matrix[v][u] = weight
    }
    
    // Check if edge exists
    func hasEdge(_ u: Int, _ v: Int) -> Bool {
        return matrix[u][v] != 0
    }
    
    // Get all neighbors
    func neighbors(_ v: Int) -> [Int] {
        var result = [Int]()
        for i in 0..<vertices {
            if matrix[v][i] != 0 {
                result.append(i)
            }
        }
        return result
    }
    
    func printGraph() {
        for row in matrix {
            print(row)
        }
    }
}

// Example: 4 vertices
let graph = GraphMatrix(vertices: 4)
graph.addEdge(0, 1)
graph.addEdge(0, 2)
graph.addEdge(1, 2)
graph.addEdge(2, 3)
graph.printGraph()
// [0, 1, 1, 0]
// [1, 0, 1, 0]
// [1, 1, 0, 1]
// [0, 0, 1, 0]
```

**Pros:**
- O(1) edge lookup
- Simple to implement
- Good for dense graphs

**Cons:**
- O(V²) space
- O(V) to iterate neighbors
- Wasteful for sparse graphs

---

### Graph Representation 2: Adjacency List

```swift
// Array of lists - each vertex has list of neighbors
class GraphList {
    var adjList: [[Int]]
    let vertices: Int
    
    init(vertices: Int) {
        self.vertices = vertices
        self.adjList = Array(repeating: [Int](), count: vertices)
    }
    
    // Add undirected edge
    func addEdge(_ u: Int, _ v: Int) {
        adjList[u].append(v)
        adjList[v].append(u)
    }
    
    // Add directed edge
    func addDirectedEdge(_ u: Int, _ v: Int) {
        adjList[u].append(v)
    }
    
    // Get neighbors
    func neighbors(_ v: Int) -> [Int] {
        return adjList[v]
    }
    
    // Remove edge
    func removeEdge(_ u: Int, _ v: Int) {
        adjList[u].removeAll { $0 == v }
        adjList[v].removeAll { $0 == u }
    }
    
    func printGraph() {
        for (vertex, neighbors) in adjList.enumerated() {
            print("\(vertex): \(neighbors)")
        }
    }
}

let graph2 = GraphList(vertices: 4)
graph2.addEdge(0, 1)
graph2.addEdge(0, 2)
graph2.addEdge(1, 2)
graph2.addEdge(2, 3)
graph2.printGraph()
// 0: [1, 2]
// 1: [0, 2]
// 2: [0, 1, 3]
// 3: [2]
```

**Pros:**
- O(V + E) space (optimal)
- Fast neighbor iteration
- Good for sparse graphs

**Cons:**
- O(degree) edge lookup
- Slightly more complex

---

### Graph Representation 3: Edge List

```swift
// List of all edges
struct Edge {
    let from: Int
    let to: Int
    let weight: Int?
    
    init(from: Int, to: Int, weight: Int? = nil) {
        self.from = from
        self.to = to
        self.weight = weight
    }
}

class GraphEdgeList {
    var edges: [Edge] = []
    let vertices: Int
    
    init(vertices: Int) {
        self.vertices = vertices
    }
    
    func addEdge(_ u: Int, _ v: Int, weight: Int? = nil) {
        edges.append(Edge(from: u, to: v, weight: weight))
    }
    
    func printGraph() {
        for edge in edges {
            if let w = edge.weight {
                print("\(edge.from) --\(w)-> \(edge.to)")
            } else {
                print("\(edge.from) -> \(edge.to)")
            }
        }
    }
}

let graph3 = GraphEdgeList(vertices: 4)
graph3.addEdge(0, 1, weight: 5)
graph3.addEdge(0, 2, weight: 3)
graph3.addEdge(1, 2, weight: 2)
graph3.printGraph()
// 0 --5-> 1
// 0 --3-> 2
// 1 --2-> 2
```

**Pros:**
- Simple representation
- Good for algorithms like Kruskal's
- Easy to sort edges

**Cons:**
- Inefficient neighbor lookup
- Not good for traversal

---

### Graph Representation 4: Dictionary/Hash Map

```swift
// Most flexible - can use any type as vertex
class Graph<T: Hashable> {
    var adjList: [T: [T]] = [:]
    
    func addVertex(_ vertex: T) {
        if adjList[vertex] == nil {
            adjList[vertex] = []
        }
    }
    
    func addEdge(_ from: T, _ to: T) {
        addVertex(from)
        addVertex(to)
        adjList[from]?.append(to)
        adjList[to]?.append(from)
    }
    
    func addDirectedEdge(_ from: T, _ to: T) {
        addVertex(from)
        addVertex(to)
        adjList[from]?.append(to)
    }
    
    func neighbors(_ vertex: T) -> [T] {
        return adjList[vertex] ?? []
    }
    
    func printGraph() {
        for (vertex, neighbors) in adjList {
            print("\(vertex): \(neighbors)")
        }
    }
}

// Example with strings
let socialGraph = Graph<String>()
socialGraph.addEdge("Alice", "Bob")
socialGraph.addEdge("Alice", "Charlie")
socialGraph.addEdge("Bob", "David")
socialGraph.printGraph()
// Alice: ["Bob", "Charlie"]
// Bob: ["Alice", "David"]
// Charlie: ["Alice"]
// David: ["Bob"]
```

---

### Weighted Graph

```swift
class WeightedGraph {
    typealias Edge = (vertex: Int, weight: Int)
    var adjList: [[Edge]]
    let vertices: Int
    
    init(vertices: Int) {
        self.vertices = vertices
        self.adjList = Array(repeating: [Edge](), count: vertices)
    }
    
    func addEdge(_ u: Int, _ v: Int, _ weight: Int) {
        adjList[u].append((v, weight))
        adjList[v].append((u, weight))
    }
    
    func addDirectedEdge(_ u: Int, _ v: Int, _ weight: Int) {
        adjList[u].append((v, weight))
    }
    
    func neighbors(_ v: Int) -> [Edge] {
        return adjList[v]
    }
    
    func printGraph() {
        for (vertex, edges) in adjList.enumerated() {
            let edgeStr = edges.map { "(\($0.vertex), w:\($0.weight))" }.joined(separator: ", ")
            print("\(vertex): [\(edgeStr)]")
        }
    }
}

let wgraph = WeightedGraph(vertices: 4)
wgraph.addEdge(0, 1, 4)
wgraph.addEdge(0, 2, 1)
wgraph.addEdge(1, 3, 1)
wgraph.addEdge(2, 3, 5)
wgraph.printGraph()
// 0: [(1, w:4), (2, w:1)]
// 1: [(0, w:4), (3, w:1)]
// 2: [(0, w:1), (3, w:5)]
// 3: [(1, w:1), (2, w:5)]
```

---

### Building Graph from Input

```swift
// From edge list
func buildGraphFromEdges(_ n: Int, _ edges: [[Int]]) -> [[Int]] {
    var graph = Array(repeating: [Int](), count: n)
    
    for edge in edges {
        let u = edge[0]
        let v = edge[1]
        graph[u].append(v)
        graph[v].append(u)
    }
    
    return graph
}

// From prerequisites (directed)
func buildGraphFromPrereqs(_ n: Int, _ prerequisites: [[Int]]) -> [[Int]] {
    var graph = Array(repeating: [Int](), count: n)
    
    for prereq in prerequisites {
        let course = prereq[0]
        let prerequisite = prereq[1]
        graph[prerequisite].append(course)  // prereq → course
    }
    
    return graph
}

// Example
let edges = [[0, 1], [0, 2], [1, 2], [2, 3]]
let graph4 = buildGraphFromEdges(4, edges)
print(graph4)
// [[1, 2], [0, 2], [0, 1, 3], [2]]
```

---

### Graph Representation Comparison

| Representation | Space | Add Edge | Remove Edge | Check Edge | Get Neighbors |
|----------------|-------|----------|-------------|------------|---------------|
| Adjacency Matrix | O(V²) | O(1) | O(1) | O(1) | O(V) |
| Adjacency List | O(V+E) | O(1) | O(E) | O(degree) | O(degree) |
| Edge List | O(E) | O(1) | O(E) | O(E) | O(E) |
| Hash Map | O(V+E) | O(1) | O(degree) | O(degree) | O(degree) |

---

### Common Graph Problems Setup

```swift
// Clone Graph
class Node {
    var val: Int
    var neighbors: [Node]
    
    init(_ val: Int) {
        self.val = val
        self.neighbors = []
    }
}

// Undirected graph from edges
func buildUndirectedGraph(_ n: Int, _ edges: [[Int]]) -> [[Int]] {
    var graph = Array(repeating: [Int](), count: n)
    for edge in edges {
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    }
    return graph
}

// Directed graph from edges
func buildDirectedGraph(_ n: Int, _ edges: [[Int]]) -> [[Int]] {
    var graph = Array(repeating: [Int](), count: n)
    for edge in edges {
        graph[edge[0]].append(edge[1])
    }
    return graph
}
```

---

### Special Graph Types

#### 1. Tree
```swift
// Tree is a connected acyclic graph
// - n vertices, n-1 edges
// - Exactly one path between any two vertices
// - Often represented as parent-child relationships
```

#### 2. DAG (Directed Acyclic Graph)
```swift
// No cycles when following directed edges
// Used for: dependencies, scheduling, topological ordering
```

#### 3. Complete Graph
```swift
// Every pair of vertices has an edge
// n vertices → n(n-1)/2 edges (undirected)
```

#### 4. Bipartite Graph
```swift
// Vertices can be divided into two sets
// All edges go between sets (no edges within set)
// Can be colored with 2 colors
```

---

### Tips for Choosing Representation

1. **Dense graph (many edges)**: Adjacency matrix
2. **Sparse graph (few edges)**: Adjacency list
3. **Weighted edges**: Adjacency list with tuples
4. **Need edge sorting**: Edge list
5. **Arbitrary vertex types**: Hash map
6. **Memory critical**: Adjacency list
7. **Fast edge lookup**: Adjacency matrix

### Graph Properties to Track

```swift
struct GraphProperties {
    let vertices: Int
    let edges: Int
    let isDirected: Bool
    let isWeighted: Bool
    let isCyclic: Bool?
    let isConnected: Bool?
    
    var density: Double {
        let maxEdges = isDirected ? vertices * (vertices - 1) : vertices * (vertices - 1) / 2
        return Double(edges) / Double(maxEdges)
    }
}
```

---

**🎯 Practice Problems:**
1. Number of Islands
2. Course Schedule (I & II)
3. Word Ladder
4. Network Delay Time
5. Minimum Height Trees

<a name="chapter-23"></a>
## Chapter 23: BFS & DFS (Breadth-First Search & Depth-First Search)

BFS and DFS are the fundamental graph traversal algorithms. Mastering them unlocks countless graph problems!

---

### Depth-First Search (DFS)

**Concept**: Explore as deep as possible before backtracking.

**Use Stack** (or recursion - call stack is implicit)

```swift
// DFS using recursion
func dfsRecursive(_ graph: [[Int]], _ start: Int) -> [Int] {
    var visited = Set<Int>()
    var result = [Int]()
    
    func dfs(_ node: Int) {
        if visited.contains(node) {
            return
        }
        
        visited.insert(node)
        result.append(node)
        
        for neighbor in graph[node] {
            dfs(neighbor)
        }
    }
    
    dfs(start)
    return result
}

// DFS using stack (iterative)
func dfsIterative(_ graph: [[Int]], _ start: Int) -> [Int] {
    var visited = Set<Int>()
    var result = [Int]()
    var stack = [start]
    
    while !stack.isEmpty {
        let node = stack.removeLast()
        
        if visited.contains(node) {
            continue
        }
        
        visited.insert(node)
        result.append(node)
        
        // Add neighbors in reverse order (to maintain order)
        for neighbor in graph[node].reversed() {
            if !visited.contains(neighbor) {
                stack.append(neighbor)
            }
        }
    }
    
    return result
}

// Example
let graph = [[1, 2], [0, 3, 4], [0, 5], [1], [1], [2]]
//      0
//     / \
//    1   2
//   / \   \
//  3   4   5

print(dfsRecursive(graph, 0))   // [0, 1, 3, 4, 2, 5]
print(dfsIterative(graph, 0))   // [0, 1, 3, 4, 2, 5]
```

**DFS Properties:**
- Time: O(V + E)
- Space: O(V) for recursion stack
- Goes deep before wide
- Uses stack (LIFO)
- Natural for: paths, cycles, connectivity

---

### Breadth-First Search (BFS)

**Concept**: Explore level by level (all neighbors before going deeper).

**Use Queue**

```swift
// BFS using queue
func bfs(_ graph: [[Int]], _ start: Int) -> [Int] {
    var visited = Set<Int>()
    var result = [Int]()
    var queue = [start]
    
    visited.insert(start)
    
    while !queue.isEmpty {
        let node = queue.removeFirst()
        result.append(node)
        
        for neighbor in graph[node] {
            if !visited.contains(neighbor) {
                visited.insert(neighbor)
                queue.append(neighbor)
            }
        }
    }
    
    return result
}

// BFS with level tracking
func bfsLevels(_ graph: [[Int]], _ start: Int) -> [[Int]] {
    var visited = Set<Int>()
    var levels = [[Int]]()
    var queue = [start]
    
    visited.insert(start)
    
    while !queue.isEmpty {
        let levelSize = queue.count
        var currentLevel = [Int]()
        
        for _ in 0..<levelSize {
            let node = queue.removeFirst()
            currentLevel.append(node)
            
            for neighbor in graph[node] {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor)
                    queue.append(neighbor)
                }
            }
        }
        
        levels.append(currentLevel)
    }
    
    return levels
}

print(bfs(graph, 0))        // [0, 1, 2, 3, 4, 5]
print(bfsLevels(graph, 0))  // [[0], [1, 2], [3, 4, 5]]
```

**BFS Properties:**
- Time: O(V + E)
- Space: O(V) for queue
- Goes wide before deep
- Uses queue (FIFO)
- Natural for: shortest path, levels, distance

---

### BFS vs DFS Comparison

| Feature | BFS | DFS |
|---------|-----|-----|
| Data Structure | Queue | Stack/Recursion |
| Order | Level by level | Deep first |
| Shortest Path | ✅ Yes (unweighted) | ❌ No |
| Memory | O(width) | O(height) |
| Complete Search | ✅ Yes | ✅ Yes |
| Use Case | Levels, shortest | Paths, cycles |

---

### Classic BFS Problems

#### Problem 1: Number of Islands

```swift
// Count connected components of 1s in 2D grid
// [
//   ["1","1","0","0","0"],
//   ["1","1","0","0","0"],
//   ["0","0","1","0","0"],
//   ["0","0","0","1","1"]
// ] → 3

func numIslands(_ grid: [[Character]]) -> Int {
    guard !grid.isEmpty else { return 0 }
    
    var grid = grid
    let rows = grid.count
    let cols = grid[0].count
    var count = 0
    
    func bfs(_ r: Int, _ c: Int) {
        var queue = [(r, c)]
        grid[r][c] = "0"
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while !queue.isEmpty {
            let (row, col) = queue.removeFirst()
            
            for (dr, dc) in directions {
                let newRow = row + dr
                let newCol = col + dc
                
                if newRow >= 0 && newRow < rows && 
                   newCol >= 0 && newCol < cols &&
                   grid[newRow][newCol] == "1" {
                    queue.append((newRow, newCol))
                    grid[newRow][newCol] = "0"
                }
            }
        }
    }
    
    for r in 0..<rows {
        for c in 0..<cols {
            if grid[r][c] == "1" {
                bfs(r, c)
                count += 1
            }
        }
    }
    
    return count
}

// DFS version
func numIslandsDFS(_ grid: [[Character]]) -> Int {
    guard !grid.isEmpty else { return 0 }
    
    var grid = grid
    let rows = grid.count
    let cols = grid[0].count
    var count = 0
    
    func dfs(_ r: Int, _ c: Int) {
        if r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != "1" {
            return
        }
        
        grid[r][c] = "0"
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    }
    
    for r in 0..<rows {
        for c in 0..<cols {
            if grid[r][c] == "1" {
                dfs(r, c)
                count += 1
            }
        }
    }
    
    return count
}
```

---

#### Problem 2: Shortest Path in Binary Matrix

```swift
// Find shortest path from (0,0) to (n-1,n-1) in 0s
// Can move in 8 directions
func shortestPathBinaryMatrix(_ grid: [[Int]]) -> Int {
    let n = grid.count
    
    guard grid[0][0] == 0 && grid[n-1][n-1] == 0 else {
        return -1
    }
    
    var grid = grid
    var queue = [(row: 0, col: 0, dist: 1)]
    grid[0][0] = 1
    
    let directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while !queue.isEmpty {
        let (r, c, dist) = queue.removeFirst()
        
        if r == n - 1 && c == n - 1 {
            return dist
        }
        
        for (dr, dc) in directions {
            let newR = r + dr
            let newC = c + dc
            
            if newR >= 0 && newR < n && newC >= 0 && newC < n && grid[newR][newC] == 0 {
                queue.append((newR, newC, dist + 1))
                grid[newR][newC] = 1
            }
        }
    }
    
    return -1
}
```

---

#### Problem 3: Word Ladder

```swift
// Transform beginWord to endWord, changing one letter at a time
// Each intermediate word must be in wordList
// "hit" → "cog" via ["hot","dot","dog","lot","log","cog"] → 5

func ladderLength(_ beginWord: String, _ endWord: String, _ wordList: [String]) -> Int {
    let wordSet = Set(wordList)
    
    guard wordSet.contains(endWord) else {
        return 0
    }
    
    var queue = [(word: beginWord, steps: 1)]
    var visited = Set([beginWord])
    
    while !queue.isEmpty {
        let (word, steps) = queue.removeFirst()
        
        if word == endWord {
            return steps
        }
        
        // Try all possible one-letter changes
        var chars = Array(word)
        for i in 0..<chars.count {
            let original = chars[i]
            
            for c in "abcdefghijklmnopqrstuvwxyz" {
                chars[i] = c
                let newWord = String(chars)
                
                if wordSet.contains(newWord) && !visited.contains(newWord) {
                    visited.insert(newWord)
                    queue.append((newWord, steps + 1))
                }
            }
            
            chars[i] = original
        }
    }
    
    return 0
}

print(ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]))  // 5
```

---

#### Problem 4: Rotting Oranges

```swift
// Fresh oranges adjacent to rotten become rotten each minute
// Return minimum minutes until all fresh are rotten (or -1 if impossible)
func orangesRotting(_ grid: [[Int]]) -> Int {
    var grid = grid
    let rows = grid.count
    let cols = grid[0].count
    var queue = [(Int, Int)]()
    var fresh = 0
    
    // Find all rotten oranges and count fresh
    for r in 0..<rows {
        for c in 0..<cols {
            if grid[r][c] == 2 {
                queue.append((r, c))
            } else if grid[r][c] == 1 {
                fresh += 1
            }
        }
    }
    
    guard fresh > 0 else { return 0 }
    
    var minutes = 0
    let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while !queue.isEmpty {
        let size = queue.count
        var rotted = false
        
        for _ in 0..<size {
            let (r, c) = queue.removeFirst()
            
            for (dr, dc) in directions {
                let newR = r + dr
                let newC = c + dc
                
                if newR >= 0 && newR < rows && newC >= 0 && newC < cols && 
                   grid[newR][newC] == 1 {
                    grid[newR][newC] = 2
                    fresh -= 1
                    queue.append((newR, newC))
                    rotted = true
                }
            }
        }
        
        if rotted {
            minutes += 1
        }
    }
    
    return fresh == 0 ? minutes : -1
}
```

---

### Classic DFS Problems

#### Problem 1: Clone Graph

```swift
class Node {
    var val: Int
    var neighbors: [Node]
    
    init(_ val: Int) {
        self.val = val
        self.neighbors = []
    }
}

func cloneGraph(_ node: Node?) -> Node? {
    guard let node = node else { return nil }
    
    var cloned = [Int: Node]()
    
    func dfs(_ original: Node) -> Node {
        if let clone = cloned[original.val] {
            return clone
        }
        
        let clone = Node(original.val)
        cloned[original.val] = clone
        
        for neighbor in original.neighbors {
            clone.neighbors.append(dfs(neighbor))
        }
        
        return clone
    }
    
    return dfs(node)
}
```

---

#### Problem 2: Course Schedule (Cycle Detection)

```swift
// Can finish all courses given prerequisites?
// numCourses = 2, prerequisites = [[1,0]] → true
// (take course 0 before course 1)

func canFinish(_ numCourses: Int, _ prerequisites: [[Int]]) -> Bool {
    var graph = Array(repeating: [Int](), count: numCourses)
    
    for prereq in prerequisites {
        let course = prereq[0]
        let pre = prereq[1]
        graph[pre].append(course)
    }
    
    var state = Array(repeating: 0, count: numCourses)
    // 0 = unvisited, 1 = visiting, 2 = visited
    
    func hasCycle(_ course: Int) -> Bool {
        if state[course] == 1 {
            return true  // Found cycle
        }
        
        if state[course] == 2 {
            return false  // Already processed
        }
        
        state[course] = 1  // Mark as visiting
        
        for neighbor in graph[course] {
            if hasCycle(neighbor) {
                return true
            }
        }
        
        state[course] = 2  // Mark as visited
        return false
    }
    
    for course in 0..<numCourses {
        if hasCycle(course) {
            return false
        }
    }
    
    return true
}

print(canFinish(2, [[1, 0]]))     // true
print(canFinish(2, [[1, 0], [0, 1]]))  // false (cycle)
```

---

#### Problem 3: Pacific Atlantic Water Flow

```swift
// Water flows to ocean if cell height <= neighbor
// Find cells that can flow to both oceans
func pacificAtlantic(_ heights: [[Int]]) -> [[Int]] {
    guard !heights.isEmpty else { return [] }
    
    let rows = heights.count
    let cols = heights[0].count
    
    var pacific = Array(repeating: Array(repeating: false, count: cols), count: rows)
    var atlantic = Array(repeating: Array(repeating: false, count: cols), count: rows)
    
    func dfs(_ r: Int, _ c: Int, _ ocean: inout [[Bool]], _ prevHeight: Int) {
        if r < 0 || r >= rows || c < 0 || c >= cols || 
           ocean[r][c] || heights[r][c] < prevHeight {
            return
        }
        
        ocean[r][c] = true
        
        dfs(r + 1, c, &ocean, heights[r][c])
        dfs(r - 1, c, &ocean, heights[r][c])
        dfs(r, c + 1, &ocean, heights[r][c])
        dfs(r, c - 1, &ocean, heights[r][c])
    }
    
    // Start from ocean edges
    for r in 0..<rows {
        dfs(r, 0, &pacific, Int.min)
        dfs(r, cols - 1, &atlantic, Int.min)
    }
    
    for c in 0..<cols {
        dfs(0, c, &pacific, Int.min)
        dfs(rows - 1, c, &atlantic, Int.min)
    }
    
    // Find cells reachable by both
    var result = [[Int]]()
    for r in 0..<rows {
        for c in 0..<cols {
            if pacific[r][c] && atlantic[r][c] {
                result.append([r, c])
            }
        }
    }
    
    return result
}
```

---

#### Problem 4: Number of Connected Components

```swift
// Count connected components in undirected graph
func countComponents(_ n: Int, _ edges: [[Int]]) -> Int {
    var graph = Array(repeating: [Int](), count: n)
    
    for edge in edges {
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    }
    
    var visited = Set<Int>()
    var count = 0
    
    func dfs(_ node: Int) {
        if visited.contains(node) {
            return
        }
        
        visited.insert(node)
        
        for neighbor in graph[node] {
            dfs(neighbor)
        }
    }
    
    for node in 0..<n {
        if !visited.contains(node) {
            dfs(node)
            count += 1
        }
    }
    
    return count
}

print(countComponents(5, [[0, 1], [1, 2], [3, 4]]))  // 2
```

---

#### Problem 5: All Paths from Source to Target

```swift
// Find all paths from node 0 to node n-1 in DAG
func allPathsSourceTarget(_ graph: [[Int]]) -> [[Int]] {
    var result = [[Int]]()
    var path = [Int]()
    let target = graph.count - 1
    
    func dfs(_ node: Int) {
        path.append(node)
        
        if node == target {
            result.append(path)
        } else {
            for neighbor in graph[node] {
                dfs(neighbor)
            }
        }
        
        path.removeLast()  // Backtrack
    }
    
    dfs(0)
    return result
}

print(allPathsSourceTarget([[1, 2], [3], [3], []]))
// [[0, 1, 3], [0, 2, 3]]
```

---

### BFS/DFS Patterns

| Pattern | Technique | Example |
|---------|-----------|---------|
| **Connected Components** | DFS/BFS from unvisited | Number of islands |
| **Shortest Path** | BFS | Word ladder |
| **Cycle Detection** | DFS with states | Course schedule |
| **All Paths** | DFS with backtracking | All paths source to target |
| **Level Order** | BFS with level tracking | Rotting oranges |
| **Flood Fill** | DFS/BFS | Pacific Atlantic |
| **Topological Sort** | DFS/BFS | Course schedule II |

---

### When to Use BFS vs DFS

**Use BFS when:**
- ✅ Need shortest path (unweighted)
- ✅ Level-by-level processing
- ✅ Distance calculation
- ✅ Closest/nearest problems

**Use DFS when:**
- ✅ Need all paths
- ✅ Cycle detection
- ✅ Connected components
- ✅ Memory-constrained (narrow graphs)
- ✅ Backtracking problems

---

### Tips & Tricks

1. **Mark visited early**: Prevent duplicate processing
2. **Restore state**: Use backtracking when needed
3. **Multi-source BFS**: Start with all sources in queue
4. **Bidirectional BFS**: Search from both ends
5. **Color coding**: Use states (white/gray/black)
6. **Level tracking**: Track distance/depth
7. **Grid as graph**: Treat 2D grid as graph

### Common Mistakes

❌ Forgetting to mark as visited  
❌ Not handling disconnected graphs  
❌ Marking visited at wrong time (BFS)  
❌ Not backtracking in DFS  
❌ Wrong boundary checks in grids  
❌ Using wrong data structure (stack vs queue)  

---

<a name="chapter-24"></a>
## Chapter 24: Advanced Graph Algorithms

Beyond basic traversal, these algorithms solve complex graph problems!

---

### Topological Sort

**Definition**: Linear ordering of vertices in DAG where for every edge u→v, u comes before v.

**Use Cases**: Task scheduling, dependency resolution, build systems

```swift
// DFS-based Topological Sort
func topologicalSort(_ numNodes: Int, _ edges: [[Int]]) -> [Int]? {
    var graph = Array(repeating: [Int](), count: numNodes)
    
    for edge in edges {
        graph[edge[0]].append(edge[1])
    }
    
    var result = [Int]()
    var state = Array(repeating: 0, count: numNodes)
    // 0 = unvisited, 1 = visiting, 2 = visited
    
    func dfs(_ node: Int) -> Bool {
        if state[node] == 1 {
            return false  // Cycle detected
        }
        
        if state[node] == 2 {
            return true  // Already processed
        }
        
        state[node] = 1
        
        for neighbor in graph[node] {
            if !dfs(neighbor) {
                return false
            }
        }
        
        state[node] = 2
        result.append(node)
        return true
    }
    
    for node in 0..<numNodes {
        if state[node] == 0 {
            if !dfs(node) {
                return nil  // Cycle exists
            }
        }
    }
    
    return result.reversed()
}

// Kahn's Algorithm (BFS-based)
func topologicalSortBFS(_ numNodes: Int, _ edges: [[Int]]) -> [Int]? {
    var graph = Array(repeating: [Int](), count: numNodes)
    var indegree = Array(repeating: 0, count: numNodes)
    
    for edge in edges {
        graph[edge[0]].append(edge[1])
        indegree[edge[1]] += 1
    }
    
    var queue = [Int]()
    for node in 0..<numNodes {
        if indegree[node] == 0 {
            queue.append(node)
        }
    }
    
    var result = [Int]()
    
    while !queue.isEmpty {
        let node = queue.removeFirst()
        result.append(node)
        
        for neighbor in graph[node] {
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0 {
                queue.append(neighbor)
            }
        }
    }
    
    return result.count == numNodes ? result : nil
}

print(topologicalSort(4, [[0, 1], [0, 2], [1, 3], [2, 3]]) ?? [])
// [0, 2, 1, 3] or [0, 1, 2, 3]
```

---

### Problem: Course Schedule II

```swift
// Return order to take courses
func findOrder(_ numCourses: Int, _ prerequisites: [[Int]]) -> [Int] {
    var graph = Array(repeating: [Int](), count: numCourses)
    var indegree = Array(repeating: 0, count: numCourses)
    
    for prereq in prerequisites {
        let course = prereq[0]
        let pre = prereq[1]
        graph[pre].append(course)
        indegree[course] += 1
    }
    
    var queue = [Int]()
    for course in 0..<numCourses {
        if indegree[course] == 0 {
            queue.append(course)
        }
    }
    
    var order = [Int]()
    
    while !queue.isEmpty {
        let course = queue.removeFirst()
        order.append(course)
        
        for next in graph[course] {
            indegree[next] -= 1
            if indegree[next] == 0 {
                queue.append(next)
            }
        }
    }
    
    return order.count == numCourses ? order : []
}
```

---

### Union-Find (Disjoint Set Union)

**Use Cases**: Connected components, cycle detection, dynamic connectivity

```swift
class UnionFind {
    private var parent: [Int]
    private var rank: [Int]
    private(set) var count: Int
    
    init(_ n: Int) {
        parent = Array(0..<n)
        rank = Array(repeating: 0, count: n)
        count = n
    }
    
    // Find with path compression
    func find(_ x: Int) -> Int {
        if parent[x] != x {
            parent[x] = find(parent[x])
        }
        return parent[x]
    }
    
    // Union by rank
    func union(_ x: Int, _ y: Int) -> Bool {
        let rootX = find(x)
        let rootY = find(y)
        
        if rootX == rootY {
            return false  // Already connected
        }
        
        if rank[rootX] < rank[rootY] {
            parent[rootX] = rootY
        } else if rank[rootX] > rank[rootY] {
            parent[rootY] = rootX
        } else {
            parent[rootY] = rootX
            rank[rootX] += 1
        }
        
        count -= 1
        return true
    }
    
    func isConnected(_ x: Int, _ y: Int) -> Bool {
        return find(x) == find(y)
    }
}

// Example: Number of provinces
func findCircleNum(_ isConnected: [[Int]]) -> Int {
    let n = isConnected.count
    let uf = UnionFind(n)
    
    for i in 0..<n {
        for j in (i + 1)..<n {
            if isConnected[i][j] == 1 {
                uf.union(i, j)
            }
        }
    }
    
    return uf.count
}
```

---

### Dijkstra's Algorithm (Shortest Path)

**Use Case**: Shortest path in weighted graph (non-negative weights)

```swift
func dijkstra(_ graph: [[(Int, Int)]], _ start: Int) -> [Int] {
    let n = graph.count
    var dist = Array(repeating: Int.max, count: n)
    dist[start] = 0
    
    // Min heap: (distance, node)
    var heap = [(dist: Int, node: Int)]()
    heap.append((0, start))
    
    while !heap.isEmpty {
        heap.sort { $0.dist < $1.dist }
        let (d, u) = heap.removeFirst()
        
        if d > dist[u] {
            continue
        }
        
        for (v, weight) in graph[u] {
            let newDist = dist[u] + weight
            
            if newDist < dist[v] {
                dist[v] = newDist
                heap.append((newDist, v))
            }
        }
    }
    
    return dist
}

// Network Delay Time problem
func networkDelayTime(_ times: [[Int]], _ n: Int, _ k: Int) -> Int {
    var graph = Array(repeating: [(Int, Int)](), count: n + 1)
    
    for time in times {
        let u = time[0]
        let v = time[1]
        let w = time[2]
        graph[u].append((v, w))
    }
    
    let dist = dijkstra(graph, k)
    var maxTime = 0
    
    for i in 1...n {
        if dist[i] == Int.max {
            return -1
        }
        maxTime = max(maxTime, dist[i])
    }
    
    return maxTime
}
```

---

### Bellman-Ford Algorithm

**Use Case**: Shortest path with negative weights, detect negative cycles

```swift
func bellmanFord(_ edges: [[Int]], _ n: Int, _ start: Int) -> [Int]? {
    var dist = Array(repeating: Int.max, count: n)
    dist[start] = 0
    
    // Relax edges n-1 times
    for _ in 0..<(n - 1) {
        for edge in edges {
            let u = edge[0]
            let v = edge[1]
            let weight = edge[2]
            
            if dist[u] != Int.max && dist[u] + weight < dist[v] {
                dist[v] = dist[u] + weight
            }
        }
    }
    
    // Check for negative cycles
    for edge in edges {
        let u = edge[0]
        let v = edge[1]
        let weight = edge[2]
        
        if dist[u] != Int.max && dist[u] + weight < dist[v] {
            return nil  // Negative cycle detected
        }
    }
    
    return dist
}
```

---

### Minimum Spanning Tree - Prim's Algorithm

```swift
func primMST(_ graph: [[(Int, Int)]], _ n: Int) -> Int {
    var inMST = Array(repeating: false, count: n)
    var heap = [(weight: Int, node: Int)]()
    heap.append((0, 0))
    
    var totalCost = 0
    
    while !heap.isEmpty {
        heap.sort { $0.weight < $1.weight }
        let (weight, u) = heap.removeFirst()
        
        if inMST[u] {
            continue
        }
        
        inMST[u] = true
        totalCost += weight
        
        for (v, w) in graph[u] {
            if !inMST[v] {
                heap.append((w, v))
            }
        }
    }
    
    return totalCost
}
```

---

### Minimum Spanning Tree - Kruskal's Algorithm

```swift
func kruskalMST(_ n: Int, _ edges: [[Int]]) -> Int {
    // Sort edges by weight
    let sortedEdges = edges.sorted { $0[2] < $1[2] }
    
    let uf = UnionFind(n)
    var totalCost = 0
    var edgesUsed = 0
    
    for edge in sortedEdges {
        let u = edge[0]
        let v = edge[1]
        let weight = edge[2]
        
        if uf.union(u, v) {
            totalCost += weight
            edgesUsed += 1
            
            if edgesUsed == n - 1 {
                break
            }
        }
    }
    
    return edgesUsed == n - 1 ? totalCost : -1
}
```

---

### Problem: Cheapest Flights Within K Stops

```swift
func findCheapestPrice(_ n: Int, _ flights: [[Int]], _ src: Int, _ dst: Int, _ k: Int) -> Int {
    var prices = Array(repeating: Int.max, count: n)
    prices[src] = 0
    
    // Relax edges k+1 times (k stops = k+1 edges)
    for _ in 0...k {
        var temp = prices
        
        for flight in flights {
            let from = flight[0]
            let to = flight[1]
            let price = flight[2]
            
            if prices[from] != Int.max {
                temp[to] = min(temp[to], prices[from] + price)
            }
        }
        
        prices = temp
    }
    
    return prices[dst] == Int.max ? -1 : prices[dst]
}
```

---

### Bipartite Graph Check

```swift
// Check if graph can be colored with 2 colors
func isBipartite(_ graph: [[Int]]) -> Bool {
    let n = graph.count
    var colors = Array(repeating: -1, count: n)
    
    func bfs(_ start: Int) -> Bool {
        var queue = [start]
        colors[start] = 0
        
        while !queue.isEmpty {
            let node = queue.removeFirst()
            
            for neighbor in graph[node] {
                if colors[neighbor] == -1 {
                    colors[neighbor] = 1 - colors[node]
                    queue.append(neighbor)
                } else if colors[neighbor] == colors[node] {
                    return false
                }
            }
        }
        
        return true
    }
    
    for i in 0..<n {
        if colors[i] == -1 {
            if !bfs(i) {
                return false
            }
        }
    }
    
    return true
}
```

---

### Advanced Graph Patterns

| Algorithm | Use Case | Time | When to Use |
|-----------|----------|------|-------------|
| **Topological Sort** | Dependencies | O(V+E) | DAG ordering |
| **Union-Find** | Connectivity | O(α(n)) | Dynamic connectivity |
| **Dijkstra** | Shortest path | O((V+E)log V) | Non-negative weights |
| **Bellman-Ford** | Shortest path | O(VE) | Negative weights |
| **Prim's/Kruskal's** | MST | O(E log V) | Minimum spanning tree |
| **Floyd-Warshall** | All pairs | O(V³) | Dense graphs |

### Tips & Tricks

1. **Choose right algorithm**: Match problem to algorithm
2. **Optimize with priority queue**: For Dijkstra, Prim's
3. **Path compression**: Essential for Union-Find
4. **Negative cycles**: Use Bellman-Ford to detect
5. **Edge relaxation**: Core concept in shortest path
6. **Greedy selection**: MST algorithms are greedy
7. **Bipartite = 2-colorable**: Check with BFS/DFS

---

**🎯 Practice Problems:**
1. Reconstruct Itinerary
2. Alien Dictionary
3. Swim in Rising Water
4. Critical Connections in Network
5. Accounts Merge

<a name="chapter-25"></a>
## Chapter 25: Sorting Algorithms

Sorting is fundamental! Understanding different algorithms helps you choose the right one and optimize performance.

---

### Sorting Algorithms Overview

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | ❌ |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | ✅ |
| Radix Sort | O(nk) | O(nk) | O(nk) | O(n+k) | ✅ |

**Stable**: Preserves relative order of equal elements

---

### 1. Bubble Sort

**Concept**: Repeatedly swap adjacent elements if they're in wrong order.

```swift
func bubbleSort(_ arr: inout [Int]) {
    let n = arr.count
    
    for i in 0..<n {
        var swapped = false
        
        for j in 0..<(n - i - 1) {
            if arr[j] > arr[j + 1] {
                arr.swapAt(j, j + 1)
                swapped = true
            }
        }
        
        // If no swaps, array is sorted
        if !swapped {
            break
        }
    }
}

var arr1 = [64, 34, 25, 12, 22, 11, 90]
bubbleSort(&arr1)
print(arr1)  // [11, 12, 22, 25, 34, 64, 90]
```

**When to use**: Small datasets, nearly sorted data  
**Pros**: Simple, stable, in-place  
**Cons**: Very slow for large datasets

---

### 2. Selection Sort

**Concept**: Find minimum element and place it at beginning.

```swift
func selectionSort(_ arr: inout [Int]) {
    let n = arr.count
    
    for i in 0..<n {
        var minIndex = i
        
        // Find minimum in remaining array
        for j in (i + 1)..<n {
            if arr[j] < arr[minIndex] {
                minIndex = j
            }
        }
        
        // Swap with current position
        if minIndex != i {
            arr.swapAt(i, minIndex)
        }
    }
}

var arr2 = [64, 25, 12, 22, 11]
selectionSort(&arr2)
print(arr2)  // [11, 12, 22, 25, 64]
```

**When to use**: Small datasets, memory constrained  
**Pros**: Simple, in-place, minimizes swaps  
**Cons**: Always O(n²), not stable

---

### 3. Insertion Sort

**Concept**: Build sorted array one element at a time.

```swift
func insertionSort(_ arr: inout [Int]) {
    for i in 1..<arr.count {
        let key = arr[i]
        var j = i - 1
        
        // Shift elements greater than key
        while j >= 0 && arr[j] > key {
            arr[j + 1] = arr[j]
            j -= 1
        }
        
        arr[j + 1] = key
    }
}

var arr3 = [12, 11, 13, 5, 6]
insertionSort(&arr3)
print(arr3)  // [5, 6, 11, 12, 13]
```

**When to use**: Small or nearly sorted arrays  
**Pros**: Simple, stable, adaptive (fast for sorted)  
**Cons**: O(n²) for random data

---

### 4. Merge Sort

**Concept**: Divide and conquer - split, sort, merge.

```swift
func mergeSort(_ arr: [Int]) -> [Int] {
    guard arr.count > 1 else { return arr }
    
    let mid = arr.count / 2
    let left = mergeSort(Array(arr[0..<mid]))
    let right = mergeSort(Array(arr[mid...]))
    
    return merge(left, right)
}

func merge(_ left: [Int], _ right: [Int]) -> [Int] {
    var result = [Int]()
    var i = 0, j = 0
    
    while i < left.count && j < right.count {
        if left[i] <= right[j] {
            result.append(left[i])
            i += 1
        } else {
            result.append(right[j])
            j += 1
        }
    }
    
    result.append(contentsOf: left[i...])
    result.append(contentsOf: right[j...])
    
    return result
}

let arr4 = [38, 27, 43, 3, 9, 82, 10]
print(mergeSort(arr4))  // [3, 9, 10, 27, 38, 43, 82]
```

**When to use**: Large datasets, need stable sort, guaranteed O(n log n)  
**Pros**: Stable, predictable performance  
**Cons**: O(n) extra space

---

### 5. Quick Sort

**Concept**: Pick pivot, partition around it, recurse.

```swift
func quickSort(_ arr: inout [Int], _ low: Int, _ high: Int) {
    if low < high {
        let pivot = partition(&arr, low, high)
        quickSort(&arr, low, pivot - 1)
        quickSort(&arr, pivot + 1, high)
    }
}

func partition(_ arr: inout [Int], _ low: Int, _ high: Int) -> Int {
    let pivot = arr[high]
    var i = low - 1
    
    for j in low..<high {
        if arr[j] <= pivot {
            i += 1
            arr.swapAt(i, j)
        }
    }
    
    arr.swapAt(i + 1, high)
    return i + 1
}

var arr5 = [10, 7, 8, 9, 1, 5]
quickSort(&arr5, 0, arr5.count - 1)
print(arr5)  // [1, 5, 7, 8, 9, 10]

// Simplified version (not in-place)
func quickSortSimple(_ arr: [Int]) -> [Int] {
    guard arr.count > 1 else { return arr }
    
    let pivot = arr[arr.count / 2]
    let less = arr.filter { $0 < pivot }
    let equal = arr.filter { $0 == pivot }
    let greater = arr.filter { $0 > pivot }
    
    return quickSortSimple(less) + equal + quickSortSimple(greater)
}
```

**When to use**: General purpose, large datasets  
**Pros**: Fast average case, in-place  
**Cons**: Worst case O(n²), not stable

---

### 6. Heap Sort

**Concept**: Build max heap, repeatedly extract maximum.

```swift
func heapSort(_ arr: inout [Int]) {
    let n = arr.count
    
    // Build max heap
    for i in stride(from: n / 2 - 1, through: 0, by: -1) {
        heapify(&arr, n, i)
    }
    
    // Extract elements from heap
    for i in stride(from: n - 1, through: 1, by: -1) {
        arr.swapAt(0, i)
        heapify(&arr, i, 0)
    }
}

func heapify(_ arr: inout [Int], _ n: Int, _ i: Int) {
    var largest = i
    let left = 2 * i + 1
    let right = 2 * i + 2
    
    if left < n && arr[left] > arr[largest] {
        largest = left
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right
    }
    
    if largest != i {
        arr.swapAt(i, largest)
        heapify(&arr, n, largest)
    }
}

var arr6 = [12, 11, 13, 5, 6, 7]
heapSort(&arr6)
print(arr6)  // [5, 6, 7, 11, 12, 13]
```

**When to use**: Guaranteed O(n log n), limited memory  
**Pros**: In-place, guaranteed performance  
**Cons**: Not stable, slower than quick sort in practice

---

### 7. Counting Sort

**Concept**: Count occurrences, calculate positions.

```swift
func countingSort(_ arr: [Int]) -> [Int] {
    guard let max = arr.max() else { return arr }
    
    var count = Array(repeating: 0, count: max + 1)
    
    // Count occurrences
    for num in arr {
        count[num] += 1
    }
    
    // Build sorted array
    var result = [Int]()
    for (num, freq) in count.enumerated() {
        result.append(contentsOf: Array(repeating: num, count: freq))
    }
    
    return result
}

let arr7 = [4, 2, 2, 8, 3, 3, 1]
print(countingSort(arr7))  // [1, 2, 2, 3, 3, 4, 8]
```

**When to use**: Small range of integers  
**Pros**: O(n+k), stable  
**Cons**: Only for integers, needs extra space

---

### 8. Radix Sort

**Concept**: Sort by each digit, starting from least significant.

```swift
func radixSort(_ arr: [Int]) -> [Int] {
    guard let max = arr.max() else { return arr }
    
    var result = arr
    var exp = 1
    
    while max / exp > 0 {
        result = countingSortByDigit(result, exp)
        exp *= 10
    }
    
    return result
}

func countingSortByDigit(_ arr: [Int], _ exp: Int) -> [Int] {
    var output = Array(repeating: 0, count: arr.count)
    var count = Array(repeating: 0, count: 10)
    
    // Count occurrences of digits
    for num in arr {
        let digit = (num / exp) % 10
        count[digit] += 1
    }
    
    // Calculate positions
    for i in 1..<10 {
        count[i] += count[i - 1]
    }
    
    // Build output array
    for i in stride(from: arr.count - 1, through: 0, by: -1) {
        let digit = (arr[i] / exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    }
    
    return output
}

let arr8 = [170, 45, 75, 90, 802, 24, 2, 66]
print(radixSort(arr8))  // [2, 24, 45, 66, 75, 90, 170, 802]
```

**When to use**: Large number of integers with limited digits  
**Pros**: O(nk), stable  
**Cons**: Only for integers, more complex

---

### Sorting Problems

#### Problem 1: Sort Colors (Dutch National Flag)

```swift
// Sort array of 0s, 1s, 2s in-place
func sortColors(_ nums: inout [Int]) {
    var low = 0, mid = 0, high = nums.count - 1
    
    while mid <= high {
        switch nums[mid] {
        case 0:
            nums.swapAt(low, mid)
            low += 1
            mid += 1
        case 1:
            mid += 1
        case 2:
            nums.swapAt(mid, high)
            high -= 1
        default:
            break
        }
    }
}

var colors = [2, 0, 2, 1, 1, 0]
sortColors(&colors)
print(colors)  // [0, 0, 1, 1, 2, 2]
```

---

#### Problem 2: Kth Largest Element (Quick Select)

```swift
// Find kth largest element - O(n) average
func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
    var nums = nums
    return quickSelect(&nums, 0, nums.count - 1, nums.count - k)
}

func quickSelect(_ nums: inout [Int], _ left: Int, _ right: Int, _ k: Int) -> Int {
    if left == right {
        return nums[left]
    }
    
    let pivot = partition(&nums, left, right)
    
    if k == pivot {
        return nums[k]
    } else if k < pivot {
        return quickSelect(&nums, left, pivot - 1, k)
    } else {
        return quickSelect(&nums, pivot + 1, right, k)
    }
}

func partition(_ nums: inout [Int], _ left: Int, _ right: Int) -> Int {
    let pivot = nums[right]
    var i = left
    
    for j in left..<right {
        if nums[j] <= pivot {
            nums.swapAt(i, j)
            i += 1
        }
    }
    
    nums.swapAt(i, right)
    return i
}

print(findKthLargest([3, 2, 1, 5, 6, 4], 2))  // 5
```

---

#### Problem 3: Merge Intervals

```swift
// Merge overlapping intervals
func merge(_ intervals: [[Int]]) -> [[Int]] {
    guard intervals.count > 1 else { return intervals }
    
    let sorted = intervals.sorted { $0[0] < $1[0] }
    var result = [sorted[0]]
    
    for interval in sorted[1...] {
        let last = result[result.count - 1]
        
        if interval[0] <= last[1] {
            // Overlapping - merge
            result[result.count - 1][1] = max(last[1], interval[1])
        } else {
            // Non-overlapping
            result.append(interval)
        }
    }
    
    return result
}

print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
// [[1, 6], [8, 10], [15, 18]]
```

---

### Choosing the Right Sort

```
Small array (< 50)?          → Insertion Sort
Nearly sorted?               → Insertion Sort
Need stable sort?            → Merge Sort
Limited memory?              → Heap Sort or Quick Sort
Integers, small range?       → Counting Sort
Large integers, few digits?  → Radix Sort
General purpose?             → Quick Sort
Guaranteed O(n log n)?       → Merge Sort or Heap Sort
```

---

<a name="chapter-26"></a>
## Chapter 26: Binary Search Mastery

Binary search is simple in concept but tricky in implementation. Master it to solve countless problems!

---

### Binary Search Basics

**Concept**: Repeatedly divide search space in half.

**Requirements**: Array must be sorted (or monotonic)

```swift
// Classic binary search
func binarySearch(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2  // Avoid overflow
        
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1  // Not found
}

let sorted = [1, 3, 5, 7, 9, 11, 13, 15]
print(binarySearch(sorted, 7))   // 3
print(binarySearch(sorted, 10))  // -1
```

**Time**: O(log n)  
**Space**: O(1)

---

### Binary Search Templates

#### Template 1: Find Exact Value

```swift
func binarySearchTemplate1(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}
```

---

#### Template 2: Find Leftmost/First Occurrence

```swift
// Find first occurrence of target
func findFirst(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    var result = -1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if arr[mid] == target {
            result = mid
            right = mid - 1  // Continue searching left
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

let arr = [1, 2, 2, 2, 3, 4, 5]
print(findFirst(arr, 2))  // 1
```

---

#### Template 3: Find Rightmost/Last Occurrence

```swift
func findLast(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    var result = -1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if arr[mid] == target {
            result = mid
            left = mid + 1  // Continue searching right
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

print(findLast(arr, 2))  // 3
```

---

#### Template 4: Lower Bound (First >= target)

```swift
func lowerBound(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count
    
    while left < right {
        let mid = left + (right - left) / 2
        
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}

print(lowerBound([1, 2, 4, 4, 5], 3))  // 2 (index of first >= 3)
```

---

#### Template 5: Upper Bound (First > target)

```swift
func upperBound(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count
    
    while left < right {
        let mid = left + (right - left) / 2
        
        if arr[mid] <= target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}

print(upperBound([1, 2, 4, 4, 5], 4))  // 4 (index of first > 4)
```

---

### Binary Search on Answer Space

**Key Insight**: Can binary search on the answer itself!

#### Problem 1: Square Root

```swift
// Find integer square root
func mySqrt(_ x: Int) -> Int {
    if x < 2 { return x }
    
    var left = 1
    var right = x / 2
    var result = 0
    
    while left <= right {
        let mid = left + (right - left) / 2
        let square = mid * mid
        
        if square == x {
            return mid
        } else if square < x {
            result = mid
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

print(mySqrt(8))  // 2
print(mySqrt(16)) // 4
```

---

#### Problem 2: Capacity To Ship Packages Within D Days

```swift
// Find minimum capacity to ship all packages in D days
func shipWithinDays(_ weights: [Int], _ days: Int) -> Int {
    var left = weights.max()!  // At least max weight
    var right = weights.reduce(0, +)  // All in one day
    
    while left < right {
        let mid = left + (right - left) / 2
        
        if canShip(weights, days, mid) {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return left
    
    func canShip(_ weights: [Int], _ days: Int, _ capacity: Int) -> Bool {
        var daysNeeded = 1
        var currentWeight = 0
        
        for weight in weights {
            if currentWeight + weight > capacity {
                daysNeeded += 1
                currentWeight = 0
            }
            currentWeight += weight
        }
        
        return daysNeeded <= days
    }
}

print(shipWithinDays([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  // 15
```

---

#### Problem 3: Koko Eating Bananas

```swift
// Find minimum eating speed to finish all bananas in h hours
func minEatingSpeed(_ piles: [Int], _ h: Int) -> Int {
    var left = 1
    var right = piles.max()!
    
    while left < right {
        let mid = left + (right - left) / 2
        
        if canEatAll(piles, h, mid) {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return left
    
    func canEatAll(_ piles: [Int], _ h: Int, _ speed: Int) -> Bool {
        var hours = 0
        for pile in piles {
            hours += (pile + speed - 1) / speed  // Ceiling division
        }
        return hours <= h
    }
}

print(minEatingSpeed([3, 6, 7, 11], 8))  // 4
```

---

### Binary Search on 2D Matrix

#### Problem 1: Search a 2D Matrix

```swift
// Matrix sorted row-wise and column-wise
func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
    guard !matrix.isEmpty else { return false }
    
    let rows = matrix.count
    let cols = matrix[0].count
    
    var left = 0
    var right = rows * cols - 1
    
    while left <= right {
        let mid = left + (right - left) / 2
        let row = mid / cols
        let col = mid % cols
        let value = matrix[row][col]
        
        if value == target {
            return true
        } else if value < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return false
}

let matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]
print(searchMatrix(matrix, 3))   // true
print(searchMatrix(matrix, 13))  // false
```

---

### Rotated Array Binary Search

```swift
// Search in rotated sorted array
func search(_ nums: [Int], _ target: Int) -> Int {
    var left = 0
    var right = nums.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if nums[mid] == target {
            return mid
        }
        
        // Determine which side is sorted
        if nums[left] <= nums[mid] {
            // Left side is sorted
            if nums[left] <= target && target < nums[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            // Right side is sorted
            if nums[mid] < target && target <= nums[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    
    return -1
}

print(search([4, 5, 6, 7, 0, 1, 2], 0))  // 4
print(search([4, 5, 6, 7, 0, 1, 2], 3))  // -1
```

---

### Binary Search Edge Cases

```swift
// Find peak element (arr[i] > arr[i-1] and arr[i] > arr[i+1])
func findPeakElement(_ nums: [Int]) -> Int {
    var left = 0
    var right = nums.count - 1
    
    while left < right {
        let mid = left + (right - left) / 2
        
        if nums[mid] > nums[mid + 1] {
            // Peak is on left side or mid itself
            right = mid
        } else {
            // Peak is on right side
            left = mid + 1
        }
    }
    
    return left
}

print(findPeakElement([1, 2, 3, 1]))      // 2
print(findPeakElement([1, 2, 1, 3, 5, 6, 4]))  // 5
```

---

### Binary Search Patterns

| Pattern | Key Insight | Example |
|---------|-------------|---------|
| **Classic Search** | Find target in sorted | Binary search |
| **First/Last Occurrence** | Modify condition | Find first/last |
| **Search on Answer** | Binary search result space | Capacity ship |
| **Rotated Array** | Find sorted half | Rotated search |
| **2D Matrix** | Treat as 1D | Matrix search |
| **Peak Finding** | Compare with neighbor | Find peak |
| **Minimize Maximum** | Binary search on range | Split array |

---

### Binary Search Decision Tree

```
Is array sorted?
├─ YES
│  ├─ Need exact match? → Classic binary search
│  ├─ Need first/last? → Modified binary search
│  └─ 2D matrix? → Treat as 1D
│
└─ NO
   ├─ Rotated sorted? → Rotated array search
   ├─ Find peak? → Peak finding
   └─ Can define check function?
      └─ YES → Binary search on answer
```

---

### Tips & Tricks

1. **Avoid overflow**: Use `mid = left + (right - left) / 2`
2. **Inclusive vs exclusive**: Be consistent with bounds
3. **Initialize carefully**: `right = n` vs `right = n - 1`
4. **Loop condition**: `left < right` vs `left <= right`
5. **Think monotonic**: Binary search needs monotonic property
6. **Check function**: For answer space, write clear check function
7. **Edge cases**: Empty array, single element, duplicates

### Common Mistakes

❌ Off-by-one errors in bounds  
❌ Infinite loop (wrong mid calculation)  
❌ Not handling duplicates  
❌ Forgetting to update result  
❌ Wrong comparison in rotated array  
❌ Overflow in mid calculation  

---

**🎯 Practice Problems:**
1. Find Minimum in Rotated Sorted Array
2. Search in Rotated Sorted Array II (with duplicates)
3. Median of Two Sorted Arrays
4. Split Array Largest Sum
5. Find K-th Smallest Pair Distance

<a name="chapter-27"></a>
## Chapter 27: Dynamic Programming

Dynamic Programming (DP) is about breaking problems into subproblems and storing results to avoid recomputation. It's powerful but challenging!

---

### What is Dynamic Programming?

**Two Key Properties:**
1. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times

**DP = Recursion + Memoization**

---

### DP Approaches

#### 1. Top-Down (Memoization)
- Start with original problem
- Recursively solve subproblems
- Cache results

#### 2. Bottom-Up (Tabulation)
- Start with smallest subproblems
- Build up to original problem
- Use array/table

---

### Classic DP: Fibonacci

```swift
// ❌ Naive recursion - O(2^n)
func fibRecursive(_ n: Int) -> Int {
    if n <= 1 { return n }
    return fibRecursive(n - 1) + fibRecursive(n - 2)
}

// ✅ Top-down with memoization - O(n)
func fibMemo(_ n: Int) -> Int {
    var memo = [Int: Int]()
    
    func fib(_ n: Int) -> Int {
        if n <= 1 { return n }
        
        if let cached = memo[n] {
            return cached
        }
        
        let result = fib(n - 1) + fib(n - 2)
        memo[n] = result
        return result
    }
    
    return fib(n)
}

// ✅ Bottom-up - O(n) time, O(n) space
func fibDP(_ n: Int) -> Int {
    if n <= 1 { return n }
    
    var dp = Array(repeating: 0, count: n + 1)
    dp[1] = 1
    
    for i in 2...n {
        dp[i] = dp[i - 1] + dp[i - 2]
    }
    
    return dp[n]
}

// ✅ Optimized - O(n) time, O(1) space
func fibOptimized(_ n: Int) -> Int {
    if n <= 1 { return n }
    
    var prev = 0
    var curr = 1
    
    for _ in 2...n {
        let next = prev + curr
        prev = curr
        curr = next
    }
    
    return curr
}

print(fibDP(10))  // 55
```

---

### 1D DP Problems

#### Problem 1: Climbing Stairs

```swift
// Ways to climb n stairs (can take 1 or 2 steps)
// n = 3 → 3 ways: (1,1,1), (1,2), (2,1)

func climbStairs(_ n: Int) -> Int {
    if n <= 2 { return n }
    
    var dp = Array(repeating: 0, count: n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in 3...n {
        dp[i] = dp[i - 1] + dp[i - 2]
    }
    
    return dp[n]
}

// Space optimized
func climbStairsOptimized(_ n: Int) -> Int {
    if n <= 2 { return n }
    
    var prev = 1, curr = 2
    
    for _ in 3...n {
        let next = prev + curr
        prev = curr
        curr = next
    }
    
    return curr
}

print(climbStairs(5))  // 8
```

---

#### Problem 2: House Robber

```swift
// Rob houses, can't rob adjacent houses
// [2, 7, 9, 3, 1] → 12 (2 + 9 + 1)

func rob(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    guard nums.count > 1 else { return nums[0] }
    
    var dp = Array(repeating: 0, count: nums.count)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in 2..<nums.count {
        // Either rob current + dp[i-2], or skip current (dp[i-1])
        dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])
    }
    
    return dp[nums.count - 1]
}

// Space optimized
func robOptimized(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    guard nums.count > 1 else { return nums[0] }
    
    var prev2 = nums[0]
    var prev1 = max(nums[0], nums[1])
    
    for i in 2..<nums.count {
        let curr = max(nums[i] + prev2, prev1)
        prev2 = prev1
        prev1 = curr
    }
    
    return prev1
}

print(rob([2, 7, 9, 3, 1]))  // 12
```

---

#### Problem 3: Longest Increasing Subsequence

```swift
// Find length of longest increasing subsequence
// [10, 9, 2, 5, 3, 7, 101, 18] → 4 ([2, 3, 7, 101])

func lengthOfLIS(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    
    var dp = Array(repeating: 1, count: nums.count)
    var maxLength = 1
    
    for i in 1..<nums.count {
        for j in 0..<i {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j] + 1)
            }
        }
        maxLength = max(maxLength, dp[i])
    }
    
    return maxLength
}

// O(n log n) using binary search
func lengthOfLISOptimal(_ nums: [Int]) -> Int {
    var tails = [Int]()
    
    for num in nums {
        var left = 0
        var right = tails.count
        
        while left < right {
            let mid = left + (right - left) / 2
            if tails[mid] < num {
                left = mid + 1
            } else {
                right = mid
            }
        }
        
        if left == tails.count {
            tails.append(num)
        } else {
            tails[left] = num
        }
    }
    
    return tails.count
}

print(lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))  // 4
```

---

### 2D DP Problems

#### Problem 1: Unique Paths

```swift
// Count paths from top-left to bottom-right (only right/down)
// m = 3, n = 2 → 3 paths

func uniquePaths(_ m: Int, _ n: Int) -> Int {
    var dp = Array(repeating: Array(repeating: 0, count: n), count: m)
    
    // Initialize first row and column
    for i in 0..<m {
        dp[i][0] = 1
    }
    for j in 0..<n {
        dp[0][j] = 1
    }
    
    for i in 1..<m {
        for j in 1..<n {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        }
    }
    
    return dp[m - 1][n - 1]
}

// Space optimized - O(n) space
func uniquePathsOptimized(_ m: Int, _ n: Int) -> Int {
    var dp = Array(repeating: 1, count: n)
    
    for _ in 1..<m {
        for j in 1..<n {
            dp[j] += dp[j - 1]
        }
    }
    
    return dp[n - 1]
}

print(uniquePaths(3, 7))  // 28
```

---

#### Problem 2: Minimum Path Sum

```swift
// Find path with minimum sum from top-left to bottom-right
func minPathSum(_ grid: [[Int]]) -> Int {
    let m = grid.count
    let n = grid[0].count
    var dp = grid
    
    // Initialize first row
    for j in 1..<n {
        dp[0][j] += dp[0][j - 1]
    }
    
    // Initialize first column
    for i in 1..<m {
        dp[i][0] += dp[i - 1][0]
    }
    
    for i in 1..<m {
        for j in 1..<n {
            dp[i][j] += min(dp[i - 1][j], dp[i][j - 1])
        }
    }
    
    return dp[m - 1][n - 1]
}

let grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(minPathSum(grid))  // 7 (1→3→1→1→1)
```

---

#### Problem 3: Longest Common Subsequence

```swift
// Find length of longest common subsequence
// "abcde", "ace" → 3 ("ace")

func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
    let s1 = Array(text1)
    let s2 = Array(text2)
    let m = s1.count
    let n = s2.count
    
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    
    for i in 1...m {
        for j in 1...n {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }
    
    return dp[m][n]
}

print(longestCommonSubsequence("abcde", "ace"))  // 3
```

---

#### Problem 4: Edit Distance

```swift
// Minimum operations to convert word1 to word2
// Operations: insert, delete, replace
// "horse", "ros" → 3

func minDistance(_ word1: String, _ word2: String) -> Int {
    let s1 = Array(word1)
    let s2 = Array(word2)
    let m = s1.count
    let n = s2.count
    
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    
    // Initialize base cases
    for i in 0...m {
        dp[i][0] = i
    }
    for j in 0...n {
        dp[0][j] = j
    }
    
    for i in 1...m {
        for j in 1...n {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1]
            } else {
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      // Delete
                    dp[i][j - 1],      // Insert
                    dp[i - 1][j - 1]   // Replace
                )
            }
        }
    }
    
    return dp[m][n]
}

print(minDistance("horse", "ros"))  // 3
```

---

### Knapsack Problems

#### 0/1 Knapsack

```swift
// Given weights and values, maximize value with weight limit
func knapsack(_ weights: [Int], _ values: [Int], _ capacity: Int) -> Int {
    let n = weights.count
    var dp = Array(repeating: Array(repeating: 0, count: capacity + 1), count: n + 1)
    
    for i in 1...n {
        for w in 1...capacity {
            if weights[i - 1] <= w {
                // Either take item or don't
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                )
            } else {
                dp[i][w] = dp[i - 1][w]
            }
        }
    }
    
    return dp[n][capacity]
}

let weights = [1, 3, 4, 5]
let values = [1, 4, 5, 7]
print(knapsack(weights, values, 7))  // 9
```

---

#### Problem: Partition Equal Subset Sum

```swift
// Check if array can be partitioned into two equal sum subsets
// [1, 5, 11, 5] → true ([1, 5, 5] and [11])

func canPartition(_ nums: [Int]) -> Bool {
    let sum = nums.reduce(0, +)
    
    guard sum % 2 == 0 else { return false }
    
    let target = sum / 2
    var dp = Array(repeating: false, count: target + 1)
    dp[0] = true
    
    for num in nums {
        for i in stride(from: target, through: num, by: -1) {
            dp[i] = dp[i] || dp[i - num]
        }
    }
    
    return dp[target]
}

print(canPartition([1, 5, 11, 5]))  // true
```

---

### Advanced DP Problems

#### Problem 1: Coin Change

```swift
// Minimum coins to make amount
// coins = [1, 2, 5], amount = 11 → 3 (5 + 5 + 1)

func coinChange(_ coins: [Int], _ amount: Int) -> Int {
    var dp = Array(repeating: amount + 1, count: amount + 1)
    dp[0] = 0
    
    for i in 1...amount {
        for coin in coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i - coin] + 1)
            }
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount]
}

print(coinChange([1, 2, 5], 11))  // 3
```

---

#### Problem 2: Word Break

```swift
// Check if string can be segmented into dictionary words
// s = "leetcode", dict = ["leet", "code"] → true

func wordBreak(_ s: String, _ wordDict: [String]) -> Bool {
    let wordSet = Set(wordDict)
    let chars = Array(s)
    let n = chars.count
    var dp = Array(repeating: false, count: n + 1)
    dp[0] = true
    
    for i in 1...n {
        for j in 0..<i {
            let substring = String(chars[j..<i])
            if dp[j] && wordSet.contains(substring) {
                dp[i] = true
                break
            }
        }
    }
    
    return dp[n]
}

print(wordBreak("leetcode", ["leet", "code"]))  // true
```

---

#### Problem 3: Decode Ways

```swift
// Count ways to decode string (1-26 = A-Z)
// "226" → 3 ("BZ", "VF", "BBF")

func numDecodings(_ s: String) -> Int {
    guard !s.isEmpty && s.first != "0" else { return 0 }
    
    let chars = Array(s)
    let n = chars.count
    var dp = Array(repeating: 0, count: n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in 2...n {
        let oneDigit = Int(String(chars[i - 1]))!
        let twoDigits = Int(String(chars[i - 2..<i]))!
        
        if oneDigit >= 1 {
            dp[i] += dp[i - 1]
        }
        
        if twoDigits >= 10 && twoDigits <= 26 {
            dp[i] += dp[i - 2]
        }
    }
    
    return dp[n]
}

print(numDecodings("226"))  // 3
```

---

### DP Patterns

| Pattern | Characteristic | Example |
|---------|---------------|---------|
| **Linear DP** | 1D array, sequence | Climbing stairs, House robber |
| **Grid DP** | 2D array, paths | Unique paths, Min path sum |
| **String DP** | Compare chars | LCS, Edit distance |
| **Knapsack** | Include/exclude | 0/1 knapsack, Partition |
| **Subsequence** | LIS-like | LIS, LCS |
| **State Machine** | Finite states | Stock problems |

---

### DP Problem-Solving Framework

```
1. Define the state
   - What does dp[i] represent?
   - What are dimensions?

2. Find recurrence relation
   - How to compute dp[i] from previous states?
   - What are the transitions?

3. Initialize base cases
   - What are trivial cases?
   - dp[0] = ?

4. Determine order of computation
   - Bottom-up or top-down?
   - Which direction to iterate?

5. Optimize space (if needed)
   - Can reduce dimensions?
   - Only need previous row/values?
```

---

### Tips & Tricks

1. **Start with recursion**: Write recursive solution first
2. **Identify subproblems**: What repeats?
3. **Draw the table**: Visualize dp array
4. **Check dimensions**: 1D or 2D?
5. **Space optimization**: Often can reduce by one dimension
6. **Edge cases**: Empty input, single element
7. **Print dp table**: Debug by printing intermediate results

---

<a name="chapter-28"></a>
## Chapter 28: Backtracking

Backtracking explores all possibilities by building candidates incrementally and abandoning them when they fail!

---

### What is Backtracking?

**Concept**: Try all possibilities, backtrack when dead end.

**Template**:
```swift
func backtrack(_ params) {
    if isComplete() {
        saveResult()
        return
    }
    
    for choice in choices {
        make(choice)
        backtrack(newParams)
        undo(choice)  // Backtrack!
    }
}
```

---

### Classic Backtracking Problems

#### Problem 1: Permutations

```swift
// Generate all permutations
// [1, 2, 3] → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

func permute(_ nums: [Int]) -> [[Int]] {
    var result = [[Int]]()
    var current = [Int]()
    var used = Array(repeating: false, count: nums.count)
    
    func backtrack() {
        if current.count == nums.count {
            result.append(current)
            return
        }
        
        for i in 0..<nums.count {
            if used[i] {
                continue
            }
            
            // Make choice
            current.append(nums[i])
            used[i] = true
            
            backtrack()
            
            // Undo choice
            current.removeLast()
            used[i] = false
        }
    }
    
    backtrack()
    return result
}

print(permute([1, 2, 3]))
```

---

#### Problem 2: Combinations

```swift
// Find all combinations of k numbers from 1 to n
// n = 4, k = 2 → [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

func combine(_ n: Int, _ k: Int) -> [[Int]] {
    var result = [[Int]]()
    var current = [Int]()
    
    func backtrack(_ start: Int) {
        if current.count == k {
            result.append(current)
            return
        }
        
        for i in start...n {
            current.append(i)
            backtrack(i + 1)
            current.removeLast()
        }
    }
    
    backtrack(1)
    return result
}

print(combine(4, 2))
```

---

#### Problem 3: Subsets

```swift
// Generate all subsets (power set)
// [1, 2, 3] → [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]

func subsets(_ nums: [Int]) -> [[Int]] {
    var result = [[Int]]()
    var current = [Int]()
    
    func backtrack(_ start: Int) {
        result.append(current)
        
        for i in start..<nums.count {
            current.append(nums[i])
            backtrack(i + 1)
            current.removeLast()
        }
    }
    
    backtrack(0)
    return result
}

print(subsets([1, 2, 3]))
```

---

#### Problem 4: Combination Sum

```swift
// Find combinations that sum to target (can reuse)
// candidates = [2, 3, 6, 7], target = 7 → [[2,2,3], [7]]

func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
    var result = [[Int]]()
    var current = [Int]()
    
    func backtrack(_ start: Int, _ remaining: Int) {
        if remaining == 0 {
            result.append(current)
            return
        }
        
        if remaining < 0 {
            return
        }
        
        for i in start..<candidates.count {
            current.append(candidates[i])
            backtrack(i, remaining - candidates[i])  // i, not i+1 (can reuse)
            current.removeLast()
        }
    }
    
    backtrack(0, target)
    return result
}

print(combinationSum([2, 3, 6, 7], 7))
```

---

#### Problem 5: N-Queens

```swift
// Place n queens on n×n board so none attack each other
func solveNQueens(_ n: Int) -> [[String]] {
    var result = [[String]]()
    var board = Array(repeating: Array(repeating: ".", count: n), count: n)
    
    func isValid(_ row: Int, _ col: Int) -> Bool {
        // Check column
        for i in 0..<row {
            if board[i][col] == "Q" {
                return false
            }
        }
        
        // Check diagonal
        var i = row - 1
        var j = col - 1
        while i >= 0 && j >= 0 {
            if board[i][j] == "Q" {
                return false
            }
            i -= 1
            j -= 1
        }
        
        // Check anti-diagonal
        i = row - 1
        j = col + 1
        while i >= 0 && j < n {
            if board[i][j] == "Q" {
                return false
            }
            i -= 1
            j += 1
        }
        
        return true
    }
    
    func backtrack(_ row: Int) {
        if row == n {
            result.append(board.map { String($0) })
            return
        }
        
        for col in 0..<n {
            if isValid(row, col) {
                board[row][col] = "Q"
                backtrack(row + 1)
                board[row][col] = "."
            }
        }
    }
    
    backtrack(0)
    return result
}

print(solveNQueens(4).count)  // 2 solutions
```

---

#### Problem 6: Sudoku Solver

```swift
// Solve Sudoku puzzle
func solveSudoku(_ board: inout [[Character]]) {
    solve(&board)
}

func solve(_ board: inout [[Character]]) -> Bool {
    for i in 0..<9 {
        for j in 0..<9 {
            if board[i][j] == "." {
                for char in "123456789" {
                    if isValid(board, i, j, char) {
                        board[i][j] = char
                        
                        if solve(&board) {
                            return true
                        }
                        
                        board[i][j] = "."  // Backtrack
                    }
                }
                return false
            }
        }
    }
    return true
}

func isValid(_ board: [[Character]], _ row: Int, _ col: Int, _ char: Character) -> Bool {
    for i in 0..<9 {
        // Check row
        if board[row][i] == char {
            return false
        }
        
        // Check column
        if board[i][col] == char {
            return false
        }
        
        // Check 3x3 box
        let boxRow = 3 * (row / 3) + i / 3
        let boxCol = 3 * (col / 3) + i % 3
        if board[boxRow][boxCol] == char {
            return false
        }
    }
    
    return true
}
```

---

#### Problem 7: Palindrome Partitioning

```swift
// Partition string into all palindromes
// "aab" → [["a","a","b"], ["aa","b"]]

func partition(_ s: String) -> [[String]] {
    var result = [[String]]()
    var current = [String]()
    let chars = Array(s)
    
    func isPalindrome(_ start: Int, _ end: Int) -> Bool {
        var left = start
        var right = end
        
        while left < right {
            if chars[left] != chars[right] {
                return false
            }
            left += 1
            right -= 1
        }
        
        return true
    }
    
    func backtrack(_ start: Int) {
        if start == chars.count {
            result.append(current)
            return
        }
        
        for end in start..<chars.count {
            if isPalindrome(start, end) {
                current.append(String(chars[start...end]))
                backtrack(end + 1)
                current.removeLast()
            }
        }
    }
    
    backtrack(0)
    return result
}

print(partition("aab"))
```

---

#### Problem 8: Letter Combinations of Phone Number

```swift
// Map phone digits to letters
// "23" → ["ad","ae","af","bd","be","bf","cd","ce","cf"]

func letterCombinations(_ digits: String) -> [String] {
    guard !digits.isEmpty else { return [] }
    
    let mapping: [Character: String] = [
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
    ]
    
    var result = [String]()
    var current = ""
    let digitArray = Array(digits)
    
    func backtrack(_ index: Int) {
        if index == digitArray.count {
            result.append(current)
            return
        }
        
        let digit = digitArray[index]
        let letters = mapping[digit]!
        
        for letter in letters {
            current.append(letter)
            backtrack(index + 1)
            current.removeLast()
        }
    }
    
    backtrack(0)
    return result
}

print(letterCombinations("23"))
```

---

#### Problem 9: Word Search

```swift
// Find if word exists in board (can move up/down/left/right)
func exist(_ board: [[Character]], _ word: String) -> Bool {
    let rows = board.count
    let cols = board[0].count
    let wordChars = Array(word)
    var board = board
    
    func backtrack(_ row: Int, _ col: Int, _ index: Int) -> Bool {
        if index == wordChars.count {
            return true
        }
        
        if row < 0 || row >= rows || col < 0 || col >= cols ||
           board[row][col] != wordChars[index] {
            return false
        }
        
        let temp = board[row][col]
        board[row][col] = "#"  // Mark as visited
        
        let found = backtrack(row + 1, col, index + 1) ||
                   backtrack(row - 1, col, index + 1) ||
                   backtrack(row, col + 1, index + 1) ||
                   backtrack(row, col - 1, index + 1)
        
        board[row][col] = temp  // Restore
        
        return found
    }
    
    for r in 0..<rows {
        for c in 0..<cols {
            if backtrack(r, c, 0) {
                return true
            }
        }
    }
    
    return false
}
```

---

### Backtracking Patterns

| Pattern | Key Feature | Example |
|---------|-------------|---------|
| **Permutations** | Order matters | Permute array |
| **Combinations** | Order doesn't matter | Combine numbers |
| **Subsets** | Include/exclude | Power set |
| **Partitioning** | Split into groups | Palindrome partition |
| **Constraint Satisfaction** | Follow rules | N-Queens, Sudoku |
| **Path Finding** | Navigate grid | Word search |

---

### Backtracking vs DP

| Feature | Backtracking | DP |
|---------|--------------|-----|
| Goal | Find all solutions | Find optimal |
| Approach | Try all paths | Build from subproblems |
| Pruning | Stop early | No pruning |
| Space | Call stack | Memo/table |
| Use Case | Enumerate, satisfy | Optimize, count |

---

### Tips & Tricks

1. **Define choices clearly**: What can you do at each step?
2. **Base case**: When to save/return?
3. **Pruning**: Skip invalid paths early
4. **Track state**: Use boolean array, set, or modify input
5. **Restore state**: Undo changes after recursion
6. **Avoid duplicates**: Sort + skip, or use set
7. **Draw tree**: Visualize recursion tree

### Common Mistakes

❌ Forgetting to backtrack (undo changes)  
❌ Not handling duplicates  
❌ Wrong base case  
❌ Not pruning (too slow)  
❌ Modifying shared state incorrectly  
❌ Off-by-one in loop bounds  

---

**🎯 Practice Problems:**
1. Generate Parentheses
2. Restore IP Addresses
3. Combination Sum II
4. Subsets II (with duplicates)
5. Expression Add Operators

<a name="chapter-29"></a>
## Chapter 29: Greedy Algorithms

Greedy algorithms make the locally optimal choice at each step, hoping to find a global optimum. They're fast but don't always work!

---

### What is a Greedy Algorithm?

**Concept**: At each step, make the choice that looks best right now.

**When Greedy Works:**
- Problem has **greedy choice property**: Local optimum leads to global optimum
- Problem has **optimal substructure**: Optimal solution contains optimal subproblems

**Greedy vs DP:**
- Greedy: Make choice, never reconsider
- DP: Consider all choices, build optimal solution

---

### Classic Greedy Problems

#### Problem 1: Coin Change (Greedy - doesn't always work!)

```swift
// Greedy works for standard coin systems (1, 5, 10, 25)
// But fails for arbitrary coins!
func coinChangeGreedy(_ coins: [Int], _ amount: Int) -> Int {
    var remaining = amount
    var count = 0
    let sorted = coins.sorted(by: >)
    
    for coin in sorted {
        while remaining >= coin {
            remaining -= coin
            count += 1
        }
    }
    
    return remaining == 0 ? count : -1
}

// Works: coins = [1, 5, 10, 25], amount = 41 → 4 (25 + 10 + 5 + 1)
// Fails: coins = [1, 3, 4], amount = 6 → greedy gives 3 (4+1+1), optimal is 2 (3+3)
```

---

#### Problem 2: Activity Selection / Meeting Rooms

```swift
// Maximum non-overlapping intervals
// [[1,3], [2,4], [3,5], [7,9]] → 3 ([1,3], [3,5], [7,9])

func eraseOverlapIntervals(_ intervals: [[Int]]) -> Int {
    guard intervals.count > 1 else { return 0 }
    
    // Sort by end time (greedy choice!)
    let sorted = intervals.sorted { $0[1] < $1[1] }
    
    var end = sorted[0][1]
    var count = 1
    
    for i in 1..<sorted.count {
        if sorted[i][0] >= end {
            count += 1
            end = sorted[i][1]
        }
    }
    
    return intervals.count - count
}

print(eraseOverlapIntervals([[1,2], [2,3], [3,4], [1,3]]))  // 1
```

---

#### Problem 3: Jump Game

```swift
// Check if can reach last index
// [2,3,1,1,4] → true (jump 1 step to index 1, then 3 steps to last)

func canJump(_ nums: [Int]) -> Bool {
    var maxReach = 0
    
    for i in 0..<nums.count {
        if i > maxReach {
            return false
        }
        maxReach = max(maxReach, i + nums[i])
    }
    
    return true
}

print(canJump([2, 3, 1, 1, 4]))  // true
print(canJump([3, 2, 1, 0, 4]))  // false
```

---

#### Problem 4: Jump Game II

```swift
// Minimum jumps to reach last index
// [2,3,1,1,4] → 2 (index 0 → 1 → 4)

func jump(_ nums: [Int]) -> Int {
    guard nums.count > 1 else { return 0 }
    
    var jumps = 0
    var currentEnd = 0
    var farthest = 0
    
    for i in 0..<nums.count - 1 {
        farthest = max(farthest, i + nums[i])
        
        if i == currentEnd {
            jumps += 1
            currentEnd = farthest
        }
    }
    
    return jumps
}

print(jump([2, 3, 1, 1, 4]))  // 2
```

---

#### Problem 5: Gas Station

```swift
// Find starting station to complete circuit
// gas = [1,2,3,4,5], cost = [3,4,5,1,2] → 3

func canCompleteCircuit(_ gas: [Int], _ cost: [Int]) -> Int {
    var totalGas = 0
    var totalCost = 0
    var tank = 0
    var start = 0
    
    for i in 0..<gas.count {
        totalGas += gas[i]
        totalCost += cost[i]
        tank += gas[i] - cost[i]
        
        if tank < 0 {
            // Can't start from any station before i
            start = i + 1
            tank = 0
        }
    }
    
    return totalGas >= totalCost ? start : -1
}

print(canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2]))  // 3
```

---

#### Problem 6: Best Time to Buy and Sell Stock II

```swift
// Buy and sell multiple times (max profit)
// [7,1,5,3,6,4] → 7 (buy at 1, sell at 5; buy at 3, sell at 6)

func maxProfit(_ prices: [Int]) -> Int {
    var profit = 0
    
    for i in 1..<prices.count {
        if prices[i] > prices[i - 1] {
            profit += prices[i] - prices[i - 1]
        }
    }
    
    return profit
}

print(maxProfit([7, 1, 5, 3, 6, 4]))  // 7
```

---

#### Problem 7: Partition Labels

```swift
// Partition string into max parts where each letter appears in at most one part
// "ababcbacadefegdehijhklij" → [9,7,8] ("ababcbaca", "defegde", "hijhklij")

func partitionLabels(_ s: String) -> [Int] {
    let chars = Array(s)
    var lastIndex = [Character: Int]()
    
    // Find last occurrence of each character
    for (i, char) in chars.enumerated() {
        lastIndex[char] = i
    }
    
    var result = [Int]()
    var start = 0
    var end = 0
    
    for (i, char) in chars.enumerated() {
        end = max(end, lastIndex[char]!)
        
        if i == end {
            result.append(end - start + 1)
            start = i + 1
        }
    }
    
    return result
}

print(partitionLabels("ababcbacadefegdehijhklij"))  // [9, 7, 8]
```

---

#### Problem 8: Queue Reconstruction by Height

```swift
// Reconstruct queue
// [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
// → [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

func reconstructQueue(_ people: [[Int]]) -> [[Int]] {
    // Sort by height descending, then by k ascending
    let sorted = people.sorted { 
        $0[0] == $1[0] ? $0[1] < $1[1] : $0[0] > $1[0]
    }
    
    var result = [[Int]]()
    
    for person in sorted {
        result.insert(person, at: person[1])
    }
    
    return result
}
```

---

#### Problem 9: Minimum Number of Arrows to Burst Balloons

```swift
// Minimum arrows to burst all balloons
// [[10,16], [2,8], [1,6], [7,12]] → 2

func findMinArrowShots(_ points: [[Int]]) -> Int {
    guard !points.isEmpty else { return 0 }
    
    let sorted = points.sorted { $0[1] < $1[1] }
    
    var arrows = 1
    var end = sorted[0][1]
    
    for i in 1..<sorted.count {
        if sorted[i][0] > end {
            arrows += 1
            end = sorted[i][1]
        }
    }
    
    return arrows
}

print(findMinArrowShots([[10,16], [2,8], [1,6], [7,12]]))  // 2
```

---

#### Problem 10: Task Scheduler

```swift
// Already covered in heaps, but greedy approach:
func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
    var freq = [Character: Int]()
    
    for task in tasks {
        freq[task, default: 0] += 1
    }
    
    let maxFreq = freq.values.max()!
    let maxCount = freq.values.filter { $0 == maxFreq }.count
    
    let partCount = maxFreq - 1
    let partLength = n - (maxCount - 1)
    let emptySlots = partCount * partLength
    let availableTasks = tasks.count - maxFreq * maxCount
    let idles = max(0, emptySlots - availableTasks)
    
    return tasks.count + idles
}
```

---

### Greedy Patterns

| Pattern | Strategy | Example |
|---------|----------|---------|
| **Interval Scheduling** | Sort by end time | Meeting rooms |
| **Activity Selection** | Choose earliest ending | Erase intervals |
| **Huffman Coding** | Build tree greedily | Compression |
| **Fractional Knapsack** | Sort by value/weight | Knapsack variant |
| **Minimum Spanning Tree** | Kruskal's, Prim's | Graph MST |
| **Dijkstra** | Choose closest vertex | Shortest path |

---

### When Greedy Works vs Fails

**✅ Greedy Works:**
- Interval scheduling
- Huffman coding
- Fractional knapsack
- Activity selection
- MST (Kruskal, Prim)
- Dijkstra's algorithm

**❌ Greedy Fails:**
- 0/1 Knapsack
- Longest path problem
- Arbitrary coin systems
- General optimization problems

---

### Tips & Tricks

1. **Prove correctness**: Greedy needs proof (exchange argument, induction)
2. **Sort first**: Many greedy algorithms start with sorting
3. **Local optimal ≠ global**: Verify greedy choice leads to global optimum
4. **Counter-examples**: Try to find cases where greedy fails
5. **Compare with DP**: If greedy fails, try DP
6. **Edge cases**: Empty input, single element, all same
7. **Interval problems**: Usually sort by start or end time

---

<a name="chapter-30"></a>
## Chapter 30: Bit Manipulation

Bit manipulation is powerful for optimization and solving certain problems elegantly!

---

### Bit Basics

```swift
// Binary representation
5 in binary: 0101
3 in binary: 0011

// Bitwise operators
a & b   // AND
a | b   // OR
a ^ b   // XOR
~a      // NOT
a << n  // Left shift (multiply by 2^n)
a >> n  // Right shift (divide by 2^n)
```

### Common Bit Tricks

```swift
// Check if kth bit is set
func isBitSet(_ num: Int, _ k: Int) -> Bool {
    return (num & (1 << k)) != 0
}

// Set kth bit
func setBit(_ num: Int, _ k: Int) -> Int {
    return num | (1 << k)
}

// Clear kth bit
func clearBit(_ num: Int, _ k: Int) -> Int {
    return num & ~(1 << k)
}

// Toggle kth bit
func toggleBit(_ num: Int, _ k: Int) -> Int {
    return num ^ (1 << k)
}

// Check if power of 2
func isPowerOfTwo(_ n: Int) -> Bool {
    return n > 0 && (n & (n - 1)) == 0
}

// Count set bits (Brian Kernighan's algorithm)
func countBits(_ n: Int) -> Int {
    var count = 0
    var num = n
    
    while num > 0 {
        num &= (num - 1)  // Clear rightmost set bit
        count += 1
    }
    
    return count
}

// Get rightmost set bit
func getRightmostSetBit(_ n: Int) -> Int {
    return n & -n
}

print(isBitSet(5, 0))       // true (0101, bit 0 is set)
print(isPowerOfTwo(16))     // true
print(countBits(7))         // 3 (0111)
```

---

### Classic Bit Manipulation Problems

#### Problem 1: Single Number

```swift
// Every element appears twice except one
// [2,2,1] → 1

func singleNumber(_ nums: [Int]) -> Int {
    var result = 0
    for num in nums {
        result ^= num  // XOR: a^a = 0, a^0 = a
    }
    return result
}

print(singleNumber([4, 1, 2, 1, 2]))  // 4
```

---

#### Problem 2: Single Number II

```swift
// Every element appears three times except one
// [2,2,3,2] → 3

func singleNumber2(_ nums: [Int]) -> Int {
    var ones = 0, twos = 0
    
    for num in nums {
        twos |= ones & num
        ones ^= num
        let threes = ones & twos
        ones &= ~threes
        twos &= ~threes
    }
    
    return ones
}

// Alternative: Count bits
func singleNumber2Alt(_ nums: [Int]) -> Int {
    var result = 0
    
    for i in 0..<32 {
        var sum = 0
        for num in nums {
            sum += (num >> i) & 1
        }
        result |= (sum % 3) << i
    }
    
    return result
}

print(singleNumber2([2, 2, 3, 2]))  // 3
```

---

#### Problem 3: Single Number III

```swift
// Two elements appear once, rest appear twice
// [1,2,1,3,2,5] → [3,5]

func singleNumber3(_ nums: [Int]) -> [Int] {
    // XOR all numbers
    var xor = 0
    for num in nums {
        xor ^= num
    }
    // xor = a ^ b (the two unique numbers)
    
    // Find rightmost set bit
    let rightmost = xor & -xor
    
    var a = 0, b = 0
    for num in nums {
        if (num & rightmost) != 0 {
            a ^= num
        } else {
            b ^= num
        }
    }
    
    return [a, b]
}

print(singleNumber3([1, 2, 1, 3, 2, 5]))  // [3, 5]
```

---

#### Problem 4: Number of 1 Bits (Hamming Weight)

```swift
// Count number of 1 bits
func hammingWeight(_ n: Int) -> Int {
    var count = 0
    var num = n
    
    while num > 0 {
        count += num & 1
        num >>= 1
    }
    
    return count
}

// Optimized
func hammingWeightOptimized(_ n: Int) -> Int {
    var count = 0
    var num = n
    
    while num > 0 {
        num &= num - 1
        count += 1
    }
    
    return count
}

print(hammingWeight(11))  // 3 (1011)
```

---

#### Problem 5: Reverse Bits

```swift
// Reverse bits of 32-bit unsigned integer
func reverseBits(_ n: UInt32) -> UInt32 {
    var result: UInt32 = 0
    var num = n
    
    for _ in 0..<32 {
        result <<= 1
        result |= num & 1
        num >>= 1
    }
    
    return result
}

print(reverseBits(0b00000010100101000001111010011100))
// 0b00111001011110000010100101000000
```

---

#### Problem 6: Power of Two

```swift
func isPowerOfTwo(_ n: Int) -> Bool {
    return n > 0 && (n & (n - 1)) == 0
}

print(isPowerOfTwo(16))  // true
print(isPowerOfTwo(18))  // false
```

---

#### Problem 7: Power of Four

```swift
func isPowerOfFour(_ n: Int) -> Bool {
    // Power of 2 AND only odd positions have 1s
    // 0x55555555 = 0101010101010101010101010101010101
    return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555) != 0
}

print(isPowerOfFour(16))  // true
print(isPowerOfFour(8))   // false
```

---

#### Problem 8: Sum of Two Integers (Without + -)

```swift
// Add two integers using bit manipulation
func getSum(_ a: Int, _ b: Int) -> Int {
    var a = a
    var b = b
    
    while b != 0 {
        let carry = a & b
        a = a ^ b
        b = carry << 1
    }
    
    return a
}

print(getSum(1, 2))  // 3
```

---

#### Problem 9: Missing Number

```swift
// Find missing number from 0 to n
// [3,0,1] → 2

func missingNumber(_ nums: [Int]) -> Int {
    var result = nums.count
    
    for (i, num) in nums.enumerated() {
        result ^= i ^ num
    }
    
    return result
}

// Alternative: sum formula
func missingNumberSum(_ nums: [Int]) -> Int {
    let n = nums.count
    let expectedSum = n * (n + 1) / 2
    let actualSum = nums.reduce(0, +)
    return expectedSum - actualSum
}

print(missingNumber([3, 0, 1]))  // 2
```

---

#### Problem 10: Bitwise AND of Range

```swift
// Bitwise AND of all numbers in range [left, right]
// [5, 7] → 4 (5 & 6 & 7 = 4)

func rangeBitwiseAnd(_ left: Int, _ right: Int) -> Int {
    var shift = 0
    var l = left
    var r = right
    
    // Find common prefix
    while l < r {
        l >>= 1
        r >>= 1
        shift += 1
    }
    
    return l << shift
}

print(rangeBitwiseAnd(5, 7))  // 4
```

---

#### Problem 11: Maximum XOR of Two Numbers

```swift
// Find maximum XOR of two numbers in array
// [3,10,5,25,2,8] → 28 (5 ^ 25)

func findMaximumXOR(_ nums: [Int]) -> Int {
    var max = 0
    var mask = 0
    
    for i in stride(from: 31, through: 0, by: -1) {
        mask |= (1 << i)
        var prefixes = Set<Int>()
        
        for num in nums {
            prefixes.insert(num & mask)
        }
        
        let candidate = max | (1 << i)
        
        for prefix in prefixes {
            if prefixes.contains(candidate ^ prefix) {
                max = candidate
                break
            }
        }
    }
    
    return max
}

print(findMaximumXOR([3, 10, 5, 25, 2, 8]))  // 28
```

---

### Bit Manipulation Patterns

| Pattern | Technique | Example |
|---------|-----------|---------|
| **XOR Properties** | a^a=0, a^0=a | Single number |
| **AND Properties** | Clear bits, check bits | Power of 2 |
| **OR Properties** | Set bits | Set kth bit |
| **Bit Counting** | Brian Kernighan | Count 1 bits |
| **Bit Masking** | Extract/modify bits | Subsets |
| **Two's Complement** | -n = ~n + 1 | Negative numbers |

---

### XOR Properties (Most Important!)

```swift
// 1. Self XOR is 0
a ^ a = 0

// 2. XOR with 0 is self
a ^ 0 = a

// 3. Commutative
a ^ b = b ^ a

// 4. Associative
(a ^ b) ^ c = a ^ (b ^ c)

// 5. Useful for swapping
a ^= b
b ^= a  // b = original a
a ^= b  // a = original b
```

---

### Common Bit Tricks Cheat Sheet

```swift
// Get bit
(n >> k) & 1

// Set bit
n | (1 << k)

// Clear bit
n & ~(1 << k)

// Toggle bit
n ^ (1 << k)

// Update bit to value v
(n & ~(1 << k)) | (v << k)

// Clear rightmost 1
n & (n - 1)

// Isolate rightmost 1
n & -n

// Check power of 2
n > 0 && (n & (n - 1)) == 0

// Multiply by 2
n << 1

// Divide by 2
n >> 1

// Modulo power of 2 (n % 2^k)
n & ((1 << k) - 1)

// Check if odd
n & 1

// Average of two numbers (avoiding overflow)
(a & b) + ((a ^ b) >> 1)
```

---

### Bit Manipulation for Subsets

```swift
// Generate all subsets using bits
func subsetsUsingBits(_ nums: [Int]) -> [[Int]] {
    var result = [[Int]]()
    let n = nums.count
    let totalSubsets = 1 << n  // 2^n
    
    for mask in 0..<totalSubsets {
        var subset = [Int]()
        for i in 0..<n {
            if (mask & (1 << i)) != 0 {
                subset.append(nums[i])
            }
        }
        result.append(subset)
    }
    
    return result
}

print(subsetsUsingBits([1, 2, 3]))
// [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

---

### Tips & Tricks

1. **Visualize in binary**: Write out bits on paper
2. **XOR for cancellation**: Pairs cancel out
3. **AND for clearing**: Clear specific bits
4. **OR for setting**: Set specific bits
5. **Shift for multiplication**: Left shift = ×2, right = ÷2
6. **Negative numbers**: Use two's complement
7. **Test with examples**: Try small numbers first

### Common Mistakes

❌ Forgetting operator precedence (use parentheses!)  
❌ Signed vs unsigned integers  
❌ Overflow in shift operations  
❌ Not handling negative numbers correctly  
❌ Confusing & (AND) with && (logical AND)  
❌ Wrong shift direction  

---

## 🎉 CONGRATULATIONS! 🎉

You've completed **all 30 chapters** of the comprehensive Data Structures & Algorithms guide in Swift!

### What You've Mastered:

✅ **Foundations** (Chapters 1-3): Complexity analysis, Swift fundamentals, problem-solving patterns  
✅ **Arrays & Strings** (Chapters 4-7): Deep dives, two pointers, sliding window  
✅ **Linked Lists** (Chapters 8-10): Singly, doubly, advanced problems  
✅ **Stacks & Queues** (Chapters 11-13): Implementations, problems, monotonic structures  
✅ **Hash Tables & Sets** (Chapters 14-15): Hash maps, set operations  
✅ **Trees** (Chapters 16-19): Binary trees, BST, traversals, advanced problems  
✅ **Heaps** (Chapters 20-21): Implementation, priority queue problems  
✅ **Graphs** (Chapters 22-24): Representations, BFS/DFS, advanced algorithms  
✅ **Sorting & Searching** (Chapters 25-26): All sorting algorithms, binary search mastery  
✅ **Advanced Topics** (Chapters 27-30): DP, backtracking, greedy, bit manipulation  

### Next Steps:

1. **Practice regularly** on LeetCode, HackerRank, Codeforces
2. **Solve problems by category** using this guide
3. **Review patterns** before interviews
4. **Build projects** applying these concepts
5. **Teach others** to solidify understanding

### Resources:

- **LeetCode**: 2000+ problems categorized
- **Cracking the Coding Interview**: Classic reference
- **Elements of Programming Interviews**: Advanced problems
- **AlgoExpert**: Curated problem sets
- **This guide**: Your comprehensive reference!

**Happy Coding! 🚀**






