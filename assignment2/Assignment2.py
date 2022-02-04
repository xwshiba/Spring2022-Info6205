# 1. LeetCode 35 - Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # typical binary search
        # Time Complexity: O(logN) | Space Complexity: O(1)
        l, r = 0, len(nums)
        while l < r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m
        return l # or return r is the same for [l, r)

# 2. LeetCode 540 - Single Element in a Sorted Array
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # only inspect even indexes: 0, 2, 4, ...
        # if nums[m] == nums[m+1]: continue searching the right part
        # else: continue searching the left part -> Note, we cannot exclude mid at this point!
        # return the first occurance on the left
        # note here: when using [l, r), l + 1 < r;
        # when using [l, r], l + 1 <= r
        # make sure there are at least ONE element left for binary search.
        # if only 1, [1], we just return left.
        # Time Complexity: O(logN) | Space Complexity: O(1)

        l, r = 0, len(nums) - 1
        while l + 1 <= r:
            m = l + (r - l) // 2
            if m % 2 == 1:
                m -= 1
            if nums[m] == nums[m+1]:
                l = m + 2
            else:
                r = m
        return nums[l]

# 3. LeetCode 153 - Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # update result for the left value for each of the sorted portion
        # Time Complexity: O(logN/2) -> O(logN) | Space Complexity: O(1)
        res = nums[0]

        l, r = 0, len(nums) - 1
        while l <= r:
            # if the portion is sorted
            if nums[l] < nums[r]:
                res = min(res, nums[l])
                break

            # if not sorted, then binary search
            m = l + (r - l) // 2
            res = min(res, nums[m])
            # if m in left sorted portion, we want to go right
            if nums[l] <= nums[m]:
                l = m + 1
            else:
                r = m - 1
        return res

# 4. LeetCode 253 - Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # sweeping lines
        # Time Complexity: O(N) | Space Complexity: O(2N) -> O(N)
        start, end = [], []
        for interval in intervals:
            start.append((interval[0], 1))
            end.append((interval[1], -1))

        times = start + end
        times.sort(key=lambda x: (x[0], x[1]))

        res, count = 0, 0
        for time in times:
            count += time[1]
            res = max(res, count)
        return res

# 5. LeetCode 347 - Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # bucket sort
        # Time Complexity: O(3N) -> O(N) | Space Complexity: O(1) excluding the result array
        from collections import Counter, defaultdict
        freq = Counter(nums) # nums: freq

        count = defaultdict(list) # k range: [1, len(nums)]
        for n, cnt in freq.items():
            count[cnt].append(n)

        res = []
        for i in range(len(nums), 0, -1):
            if count[i]: # if it's empty, we skip it.
                res += count[i]
                if len(res) == k:
                    return res

# 6. LeetCode 16 - 3Sum Closest
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # sort nums first
        # use two pointers to reach the result
        # if cur sum < target, increase left pointer; if cur sum > target, increase right pointer
        # overall, keep updating min_diff and min_sum
        # Time Complexity: O(NlogN + N^2) -> O(N^2) | Space Complexity: O(1)
        nums.sort()
        N = len(nums)
        min_diff = float('inf')
        min_sum = float('inf')

        for i in range(N - 2): # make sure we have 3 elements left
            j = i + 1
            k = N - 1
            while j < k:
                cur_sum = nums[i] + nums[j] + nums[k]
                cur_diff = abs(cur_sum - target)
                if cur_diff < min_diff:
                    min_diff = cur_diff
                    min_sum = cur_sum
                if cur_sum < target:
                    j += 1
                elif cur_sum >= target:
                    k -= 1
        return min_sum

# 7. LeetCode 57 - Insert Interval
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # l_not_overlapped + [updated_overlapped_Interval] + r_not_overlapped
        # Time Complexity: O(N) | Space Complexity: O(1) excluding the result array
        start, end = newInterval[0], newInterval[1]
        l, r = [], []
        for interval in intervals:
            if interval[1] < start:
                l.append(interval)
            elif interval[0] > end:
                r.append(interval)
            else:
                start = min(interval[0], start)
                end = max(interval[1], end)
        return l + [[start, end]] + r

# 8. LeetCode 435 - Non-overlapping Intervals
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # sort by the end time, we want to keep the intervals who end earlier, no matter of their start time
        # Time Complexity: O(NlogN) | Space Complexity: O(1) excluding the result array
        intervals.sort(key=lambda x: x[1])
        pre = intervals[0]
        res = 0
        for i in range(1, len(intervals)):
            if pre[1] > intervals[i][0]:
                res += 1
            else:
                pre = intervals[i]
        return res

# 9. LeetCode 986 - Interval List Intersections
# Solution 1 - Two pointers optimized for 2 lists problem
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # two pointers. if overlap, calc the overlapped area
        # check overlap - everytime, we get the max start and min end, if end-start >= 0, they overlap
        # after each check, move the pointer that ends earlier that has a larger possibility to overlap
        # Time Complexity: O(M + N) | Space Complexity: O(1) excluding the result array
        N1 = len(firstList)
        N2 = len(secondList)
        res = []
        if N1 == 0 or N2 == 0:
            return res
        i, j = 0, 0
        while i < N1 and j < N2:
            start = max(firstList[i][0], secondList[j][0])
            end = min(firstList[i][1], secondList[j][1])
            duration = end - start
            if duration >= 0:
                res.append([start, end])
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1
        return res
# Solution 2 - Min heap for scaling up to multiple lists problems
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # heap solution for larger scale
        # merge all the intervals together in heap
        # min heap already sort by start time, then end time
        # pop out two intervals for comparison
        # if end1 >= start2, there's overlap. NOTE: end1, end2 is undefined. we need to compare
        # after appending overlap areas, we push the one that ends later back to the heap to check if there are other overlaps
        # Time Complexity: O(log(M+N)) | Space Complexity: O(M + N) excluding the result array

        import heapq
        intervals = firstList + secondList
        heapq.heapify(intervals)

        res = []
        while len(intervals) > 1: # as long as there are two intervals for comparison, if only 1, then no overlap just break.
            start1, end1 = heapq.heappop(intervals)
            start2, end2 = heapq.heappop(intervals)

            if end1 >= start2:
                res.append([start2, min(end1, end2)])

            if end2 >= end1:
                heapq.heappush(intervals, [start2, end2])
            else:
                heapq.heappush(intervals, [start1, end1])

        return res

# 10. LeetCode 18 - 4Sum
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # similar to 3Sum - Two Pointers
        # sort, make sure i < j < p < q, set i, j and move p,q
        # avoid duplications for i, j and p, q
        # Time Complexity: O(N^3) | Space Complexity: O(1) excluding the result array
        nums.sort()
        N = len(nums)
        res = []
        for i in range(N-3):
            # avoid duplications
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1, N-2):
                # avoid duplications
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                p, q = j+1, N-1
                while p < q:
                    cur_sum = nums[i] + nums[j] + nums[p] + nums[q]
                    if cur_sum < target:
                        p += 1
                    elif cur_sum > target:
                        q -= 1
                    elif cur_sum == target:
                        res.append([nums[i], nums[j], nums[p], nums[q]])
                        # avoid duplications
                        while (p < q and nums[p] == nums[p+1]):
                            p += 1
                        while (p < q and nums[q] == nums[q-1]):
                            q -= 1

                        p += 1
                        q -= 1
        return res

