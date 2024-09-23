class Solution(object):
    def minMovesToSeat(self, seats, students):
        """
        :type seats: List[int]
        :type students: List[int]
        :rtype: int
        """
        seats = sorted(seats)
        students = sorted(students)
        answer = 0
        for seat, student in zip(seats, students):          
            answer += abs(student - seat)
        return answer