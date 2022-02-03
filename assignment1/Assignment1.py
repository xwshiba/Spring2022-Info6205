# info6205 Assignment 1
# 1. LeetCode 75 - Sort Colors
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # The Dutch Flag Problem
        # 1. 3 pointers, left zero, right two, and the curr pointer
        # 2. if curr == 1, continue
        # 3. if curr == 0, swap left and curr. Do NOT check curr bc must be 1, increasing curr pointer & left pointer
        # 4. if curr == 2, swap right and curr. Do not increase the curr pointer, just increasing the right pointer.
        # Time Complexity: O(N) | Space Complexity: O(1)

        if len(nums) == 1:
            return

        red = 0
        blue = len(nums) - 1
        idx = 0

        # what if idx < blue?
        # 2,0,1
        # 1,0,2 -> stopped. we still need to check blue

        while idx <= blue:  # O(N)
            if nums[idx] == 1:
                idx += 1
            elif nums[idx] == 0:
                nums[idx], nums[red] = nums[red], nums[idx]
                red += 1
                idx += 1  # the swapped back number must be 1, we do not need to check again
            elif nums[idx] == 2:
                nums[idx], nums[blue] = nums[blue], nums[idx]
                blue -= 1
                # we dont use i+= 1 because we need to check the swapped back number
        return

# 2. LeetCode 229 - Majority Element II


class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        # Boyer-Moore Voting Algorithm
        # FACTS:
        # at most one majority element which is more than ⌊n/2⌋ times.
        # at most two majority elements which are more than ⌊n/3⌋ times.
        # at most three majority elements which are more than ⌊n/4⌋ times.

        # 1. initiate two candidates, two counters
        # 2. traversing nums, if num matches either candidate, increasing
        # 3. If it doesn't match either, decreasing both counters
        # 4. if any decreased to zero, change num to the cur one
        # 5. checking order MATTERS. check match first, and check 0 later.
        # 6. second pass to ensure two candidates all > [n/2] - e.g. exception:[3,2,3]
        # Time Complexity: O(2N) -> O(N) | Space Complexity: O(1)

        if not nums:
            return None

        # prefer not to initiate cand1,cand2 to a num in nums first, if len(nums) == 1
        cand1, cand2, count1, count2 = None, None, 0, 0

        # check if num == cand first! if check count first, test case [2,2] will not pass.
        # for [2,2] both cand will be assigned. we just need return 1 here.
        for num in nums:  # O(N)
            if num == cand1:
                count1 += 1
            elif num == cand2:
                count2 += 1
            elif count1 == 0:
                count1 += 1
                cand1 = num
            elif count2 == 0:
                count2 += 1
                cand2 = num
            else:
                count1 -= 1
                count2 -= 1

        N = len(nums)
        count1, count2 = 0, 0
        for num in nums:  # O(N)
            if num == cand1:
                count1 += 1
            elif num == cand2:
                count2 += 1

        res = []
        if count1 > N // 3:
            res.append(cand1)
        if count2 > N // 3:
            res.append(cand2)
        return res
        # same as return [c for c in [cand1, cand2] if nums.count(c) > (N // 3)]

# 3. LeetCode 247. H-Index


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        # confusing concept for h-index -> easier to understand by graphics.
        # approach 1 - sort and count
        # approach 2 - counting sort
        # 1. h-index range in [0, len(arr)]
        # 2. count sort from [0, 0, ... 0] -> len(arr) + 1
        # 3. if citations[i] >= len(arr), calculate it under len(arr)
        # 4. traverse backwards to sum up
        # 5. find the first value that i <= sum[i:N+1]
        # Time Complexity: O(2N) -> O(N) | Space Complexity: O(1)

        N = len(citations)
        count = [0 for _ in range(N + 1)]  # including 0, [0,0,0]
        for i in range(N):  # O(N)
            if citations[i] > N:
                count[N] += 1
            else:
                count[citations[i]] += 1

        total = 0
        for i in range(N, -1, -1):  # O(N)
            total += count[i]
            if total >= i:
                return i

# 4. LeetCode 349. Intersection of Two Arrays


class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 1. initiate a hashmap for a smaller size nums array
        # 2. traverse the smaller size nums array, save freq to hashmap. {num: freq}
        # 3. traverse the other array
        # 4. if num in hashmap and freq > 0: append to res, reset freq to 0 -> avoid duplications
        # Time Complexity: O(M + N) -> M = len(nums1), N = len(nums2) | Space Complexity: O(min(M, N))
        import collections
        res = []
        if len(nums1) < len(nums2):
            nums1, nums2 = nums2, nums1
        n1 = collections.Counter(nums1)  # O(M)
        for num in nums2:  # O(N)
            if num in n1 and n1[num] > 0:
                res.append(num)
                n1[num] = 0
        return res

# 5. LeetCode 658. Find K Closest Elements


class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # Sorting - use min priority queue
        # Time Complexity: O(NlogN + klogk) | Space Complexity: O(N)
        import heapq
        heap = []
        for num in arr:  # O(NlogN)
            dist = abs(num - x)
            heapq.heappush(heap, (dist, num))
        res = []
        for i in range(k):
            d, n = heapq.heappop(heap)
            res.append(n)
        res.sort()  # O(klogk)
        return res

# 6. LeetCode 767. Reorganize String


class Solution:
    def reorganizeString(self, s: str) -> str:
        # Greedy Approach:
        # separate the most freq char use the second most freq char
        # using a hashmap + priority queue to pop out the most freq
        # 1. Get the freq of all the chars in s
        # 2. initiate a max heap
        # note: since py use min heap -> (-freq, char) in heap to get sorted
        # 3. pop out the most and next freq char and put in res
        # 4. decrease freq by 1 for those two chars
        # 5. if any of char freq > 1, put them back to the heap
        # 6. corner case: if max heap still has chars and freq > 1, return None
        # 7. otherwise append the last char and return
        # Time Complexity: O(NlogN) | Space Complexity: O(N)

        from collections import Counter
        import heapq
        freq = Counter(s)  # O(N), O(N)
        heap = []

        for char, cnt in freq.items():  # O(NlogN), O(N)
            heapq.heappush(heap, (-cnt, char))

        res = []
        while len(heap) > 1:  # O(NlogN)
            cnt1, curr = heapq.heappop(heap)
            cnt2, nxt = heapq.heappop(heap)
            res.append(curr)
            res.append(nxt)
            if cnt1 != -1:  # notice this is a min heap by default
                heapq.heappush(heap, (cnt1 + 1, curr))
            if cnt2 != -1:
                heapq.heappush(heap, (cnt2 + 1, nxt))

        if len(heap) > 0:
            cnt3, last = heapq.heappop(heap)
            if cnt3 < -1:
                return ""
            res.append(last)

        return "".join(res)  # O(N)

# 7. LeetCode 791. Custom Sort String


class Solution:
    def customSortString(self, order: str, s: str) -> str:
        # bucket sort
        # 1. initiate a hashmap for s: {char: freq} as s may contain duplicates
        # 2. traverse every char in order: if c in hashmap, append all chars to res, reset the freq_s[c] = 0 to avoid later duplications
        # 3. traverse hashmap, if still chars remaining, append them to the last of the res
        # Time Complexity: O(M+N) -> M: len(order), N: len(s) | Space Complexity: O(N)

        from collections import Counter
        freq_s = Counter(s)  # O(N), O(N)
        res = []

        #
        for char in order:  # O(M)
            # if c not in freq_s, Counter obj will return 0
            res.append(char * freq_s[char])
            freq_s[char] = 0

        for char in freq_s:  # O(N)
            res.append(char * freq_s[char])

        return "".join(res)  # O(N)

# 8. LeetCode 969. Pancake Sorting


class Solution:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        # Note: examples are confusing
        # brute force
        # pancake sort can only sort the first k items
        # strategy: traverse backwards -> get the largest item in arr and swap it to the the last ->
        # keep sorting the previous N-1 items
        # 1. pick the last one in the cur arr
        # 2. traverse the previous arr to see whether it's the largest in the curr arr
        # 3. if it's the max in curr arr, continue
        # 4. if max is another arr[j], max_i = j, then do pancake sorting:
        # 5. reverse arr[0] ~ arr[max_i], then reverse arr[0] ~ arr[i]
        # Time Complexity: O(N^2) | Space Complexity: O(N), since it's required, so O(1)

        def reverse(end):  # O(N)
            """reverse the arr in-place"""
            start = 0
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start, end = start + 1, end - 1
            return

        N = len(arr)
        res = []  # O(N)
        for i in range(N-1, -1, -1):  # O(N)
            max_i = i
            for j in range(i):  # O(N)
                if arr[j] > arr[max_i]:
                    max_i = j

            # if max_i not in correct place, do pancake sorting
            if max_i != i:
                reverse(max_i)
                reverse(i)
                res.append(max_i+1)
                res.append(i+1)

        return res

# 9. LeetCode 1636. Sort Array by Increasing Frequency


class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        # using min heap to sort
        # Time Complexity: O(NlogN) | Space Complexity: O(2N) -> O(N)
        import heapq
        from collections import Counter
        freq = Counter(nums)  # {num: freq}
        heap = []
        for num in freq:
            heapq.heappush(heap, (freq[num], -num))

        res = []
        while len(heap) > 0:
            cnt, num = heapq.heappop(heap)
            curr = [-num] * cnt
            res.extend(curr)

        return res

        # simpler sorting solution - using custom sorting
        # Time Complexity: O(NlogN) | Space Complexity: O(2N) -> O(N)
        # freq = Counter(nums) # {num: freq}
        # return sorted(nums, key=lambda x: (freq[x], -x))

# 10. LeetCode 692. Top K Frequent Words


class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # hashmap + priority queue
        # 1. initiate a freq map for words
        # 2. put in max heap using freq of words
        # 3. pop out the first k elements
        # Time Complexity: O(NlogN) | Space Complexity: O(2N) -> O(N)

        from collections import Counter
        import heapq
        freq = Counter(words)  # O(N), O(N)
        heap = []
        for word in freq:  # O(NlogN), O(N)
            heapq.heappush(heap, (-freq[word], word))
        res = []
        for i in range(k):  # O(k)
            cnt, word = heapq.heappop(heap)
            res.append(word)
        return res
