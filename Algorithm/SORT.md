# 알고리즘 (SORT)

## 사전 설명

- 안정 정렬 / 불안정 정렬
    
    안정 정렬: 중복된 값이 있을 때 입력 순서와 동일하게 정렬 되는 정렬
    
    - 예시) 삽입 정렬, 버블 정렬, 병합 정렬
        
        ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled.png)
        
    
    불안정 정렬: 중복된 값이 있을 때 입력 순서와 관계 없이 무작위로 섞인 상태로 정렬되는 정렬
    
    - 예시) 선택 정렬, 퀵 정렬, 힙 정렬
        
        ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%201.png)
        
- 참조 지역성의 원리 (Locality)
    
    O(nlogn) 은 *C* × *n*log*n*+*α* 이라는 의미로 *α*는 비교적 무시할 수 있는 부분이고 C에 따라 같은 nlogn이더라도 시간 차이 발생. 
    
    이 C에 영향을 미치는 요소로 참조 지역성이 있다.
    
    데이터를 캐시 메모리에서 읽어오는가 메인 메모리에서 읽어 오는가?
    
    ⇒ 최근 참조한 메모리나 인접한 메모리를 사용하는가?
    

# O(nlogn)의 대표적인 정렬

- 힙 정렬(Heap Sort)
    - 설명
        
        자료구조 “힙(heap)”: 완전 이진트리의 일종으로 우선순위 큐를 위해 만들어짐, 최댓값과 최솟값을 쉽게 구할 수 있는 자료구조이다.
        
        1. 최대힙 구조를 만들어 준다.
        2. 가장 큰 수(루트 노드)와 가장 작은 수의 위치를 바꿔준다.
        3. 힙의 크기를 하나 줄여준 뒤의 트리를 최대 힙 구조로 다시 바꿔준다. 
        4. 2번과 3번을 반복한다.
        - 예시
            
            ![Heapsort-example.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Heapsort-example.gif)
            
    - 코드
        
        ```java
        import java.util.Arrays;
        import java.util.Random;
        
        public class HeapSort {
        
            static final int N = 10;
        
            public static void main(String[] args) {
                Random random = new Random(); // 랜덤함수를 이용
        
                int[] arr = new int[N];
                for (int i = 0; i < N; i++) {
                    arr[i] = random.nextInt(100); // 0 ~ 99
                }
        
                System.out.println("정렬 전: " + Arrays.toString(arr));
                heapSort(arr);
                System.out.println("정렬 후: " + Arrays.toString(arr));
            }
        
            private static void heapSort(int[] arr) {
                int n = arr.length;
        
                // maxHeap을 구성
                // n/2-1 : 부모노드의 인덱스를 기준으로 왼쪽(i*2+1) 오른쪽(i*2+2)
                // 즉 자식노드를 갖는 노트의 최대 개수부터
                for (int i = n / 2 - 1; i >= 0; i--) {
                    heapify(arr, n, i); // 일반 배열을 힙으로 구성
                }
        
                for (int i = n - 1; i > 0; i--) {
                    swap(arr, 0, i);
                    heapify(arr, i, 0); // 요소를 제거한 뒤 다시 최대힙을 구성
                }
            }
        
            private static void heapify(int[] arr, int n, int i) {
                int p = i;
                int l = i * 2 + 1;
                int r = i * 2 + 2;
        
                // 왼쪽 자식노드
                if (l < n && arr[p] < arr[l])
                    p = l;
                // 오른쪽 자식노드
                if (r < n && arr[p] < arr[r])
                    p = r;
        
                // 부모노드 < 자식노드
                if (i != p) {
                    swap(arr, p, i);
                    heapify(arr, n, p);
                }
            }
        
            private static void swap(int[] arr, int a, int b) {
                int temp = arr[a];
                arr[a] = arr[b];
                arr[b] = temp;
            }
        }
        ```
        
    - 장점
        - 추가 적인 메모리 사용이 없음
        - 최악의 경우에도 시간복잡도가 nlogn
    - 단점
        - 불안정 정렬
        - 참조 지역성이 좋지 않음
        
        ![heap.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/heap.gif)
        
- 병합 정렬(Merge Sort)
    - 설명
        
        배열을 반으로 나누어  좌측과 우측으로 계속 나눈 뒤 나누어진 부분 내에서 정렬을 한 뒤 병합하는 정렬
        
        1. 분할(divide): 정렬되지 않은 리스트를 절반으로 잘라 비슷한 크기의 두 리스트로 나눈다.
        2. 정복(conquer): 나누어진 리스트를 재귀적으로 정렬한다.
        3. 결합(combine): 두 부분의 리스트를 다시 하나의 정렬 리스트로 합병하고 이 결과를 임시 배열에 저장한다.
        4. 복사(copy): 임시배열에 저장된 결과를 원래의 배열에 복사한다.
        - 예시)
            
            ![mergesort.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/mergesort.gif)
            
    - 코드
        
        ```java
        import java.util.Arrays;
        import java.util.Random;
        
        public class MergeSort {
        
            static final int N = 10;
        
            public static void main(String[] args) {
                Random random = new Random(); // 랜덤함수를 이용
        
                int[] arr = new int[N];
                for (int i = 0; i < N; i++) {
                    arr[i] = random.nextInt(100); // 0 ~ 99
                }
        
                System.out.println("정렬 전: " + Arrays.toString(arr));
                mergeSort(0, N - 1, arr);
                System.out.println("정렬 후: " + Arrays.toString(arr));
            }
        
            // divide
            private static void mergeSort(int start, int end, int[] arr) {
                if (start >= end)
                    return;
        
                int mid = (start + end) / 2;
                mergeSort(start, mid, arr); // left
                mergeSort(mid + 1, end, arr); // right
        
                merge(start, mid, end, arr);
            }
        
            // conquer
            private static void merge(int start, int mid, int end, int[] arr) {
                int[] temp = new int[end - start + 1];
                int i = start, j = mid + 1, k = 0;
        
                // combine
                while (i <= mid && j <= end) {
                    if (arr[i] < arr[j])
                        temp[k++] = arr[i++];
                    else
                        temp[k++] = arr[j++];
                }
                while (i <= mid)
                    temp[k++] = arr[i++];
                while (j <= end)
                    temp[k++] = arr[j++];
        
                // copy
                while (k-- > 0)
                    arr[start + k] = temp[k];
            }
        }
        ```
        
    - 장점
        - 안정 정렬
        - 최악의 경우에도 nlogn의 시간복잡도
    - 단점
        - 합병의 과정에서 추가적인 공간이 필요
- 퀵 정렬(Quick Sort)
    - 설명
        
        분할정복의 방식으로 정렬
        
        1. 리스트의 중 하나의 원소를 고르고 이를 pivot이라고 칭한다.
        2. pivot의 앞에는 pivot보다 작은 원소들이 뒤에는 큰 원소들이 오도록 하고 pivot을 기준으로 두 개의 리스트로 나눈다.
        3. 분할된 리스트에서도 1번과 2번을  리스트 크기가 0 또는 1이 될 때까지 반복한다.
        - 예시)
            
            ![Sorting_quicksort_anim.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Sorting_quicksort_anim.gif)
            
    - 코드
        
        ```java
        import java.util.Arrays;
        import java.util.Random;
        
        public class QuickSort {
            static final int N = 10;
        
            public static void main(String[] args) {
                Random random = new Random(); // 랜덤함수를 이용
        
                int[] arr = new int[N];
                for (int i = 0; i < N; i++) {
                    arr[i] = random.nextInt(100); // 0 ~ 99
                }
        
                System.out.println("정렬 전: " + Arrays.toString(arr));
                quickSort(0, N - 1, arr);
                System.out.println("정렬 후: " + Arrays.toString(arr));
            }
        
            private static void quickSort(int start, int end, int[] arr) {
                if (start >= end)
                    return;
        
                int left = start + 1, right = end;
                int pivot = arr[start];
        
                while (left <= right) {
                    while (left <= end && arr[left] <= pivot)
                        left++;
                    while (right > start && arr[right] >= pivot)
                        right--;
        
                    if (left <= right) {
                        swap(arr, left, right);
                    } else {
                        swap(arr, start, right);
                    }
                }
                quickSort(start, right - 1, arr);
                quickSort(right + 1, end, arr);
            }
        
            private static void swap(int[] arr, int a, int b) {
                int temp = arr[a];
                arr[a] = arr[b];
                arr[b] = temp;
            }
        }
        ```
        
    - 장점
        - 참조 지역성의 면에서 다른 nlogn 정렬들보다 좋아 비교적 가장 빠름.
        - 추가 메모리 공간을 필요로 하지 않음
    - 단점
        - 불안정정렬
        - 정렬된 배열일 경우 시간복잡도가 O(n^2)로 최악
        

- cf
    - merge
        
        ![비교merge.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/%EB%B9%84%EA%B5%90merge.gif)
        
    - quick
    
    ![비교quick.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/%EB%B9%84%EA%B5%90quick.gif)
    

# Tim Sort(Insertion Sort + Merge Sort)

- Tim Sort의 등장 배경
    
    C의 값이 너무 커지지 않고 추가 메모리도 많이 사용하지 않으며 최악의 경우에도 nlogn의 시간복잡도를 만족하는 정렬을 위해
    
- Tim Sort에서 삽입정렬(O(n^2))을 사용하는 이유
    
    삽입정렬은 인접한 메모리와 비교를 반복하기 때문에 참조 지역성의 원리를 매우 만족한다. n이 작을 경우 퀵정렬보다도 빠르다.
    
    ⇒ 전체를 작은 n(2^x)으로 나눠서 각각을 삽입정렬로 정렬한 뒤 병합하면 더 빠를 것이라고 예상, 단 이때 너무 작게 나누는 경우에는 병합 동작이 많이 생기므로 주로 n을 32나 64로 둔다.
    
- 설명
    1. binary insertion sort
    
    앞의 두개의 원소를 놓고 증가/감소를 정한다.
    
    이후 각각의 리스트를 정렬하는데 일반적인 삽입 정렬을 사용한다면 하나의 원소를 삽입할 때 O(n)번의 비교를 진행하지만 이진삽입 정렬을 하는 정우 O(logn)의 비교를 하므로 조금 더 시간을 절약할 수 있다. (단 지역성의 면만 고려할 경우 이진삽입정렬이 떨어진다)
    
    하나의 run이 이미 정렬되어 있을 경우 최선의 시간복잡도는 O(n)이 될 수 있다.
    
    ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%202.png)
    
    1. merge
    
    스택을 사용해 특정 조건에 만족하는 합칠 run을 찾는다.
    
    ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%203.png)
    
    위 조건을 만족할 경우 좋은 점
    
    - run의 수를 작게 유지할 수 있다.
    - 비슷한 크기의 run과 병합할 수 있다.
    
    ⇒ 최소한 의 메모리를 이용하여 최대의 효율
    
    - 만일 스택에 push할 때 위 조건이 만족되지 않을 경우
        
        만족할 때까지 인접한 두 run 중 더 비슷한 run과 병합을 진행한다.
        
        ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%204.png)
        
    
    run들을 오름 차순으로 바꾸어준다.
    
    ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%205.png)
    
    이후 두 덩어리중 작은 run을 메모리에 따로 복사해 둔다.
    
    앞의 run이 클 경우 각 run의 끝부분부터 큰 순서대로 뒤를 채우고 뒤의 run이 클 경우 각 run의 앞부분부터 작은 순서대로 앞을 채운다.
    
    ![timsortmerge1.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/timsortmerge1.gif)
    
    ![timsortmerge2.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/timsortmerge2.gif)
    
    이 때 작은 run만 메모리를 사용하므로 일반 병합정렬보다 사용하는 메모리가 적다.
    
    - 추가(galloping)
        
        galloping은 병합 과정에서 특정 횟수(k) 이상 하나의 run에서만 병합이 될 경우 진행
        
        ![galloping.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/galloping.gif)
        
        몇개의 원소들을 뛰어넘으면서 비교를 진행 하다가 더이상 선택되지 않는 부분을 만났을 경우 그 이전의 지점과 그 지점 사이의 범위에서만 이분 탐색을 진행하여 어디까지 병합할지를 정하는 것
        
        실제 사용에서는 galloping mode에 들어가는 횟수가 많을 경우 k를 감소, 아닐 경우 증가하는 형태로도 사용한다.
        
- 장점
    - 큰 수의 sort든 작은 수의 sort든 빠른 속도를 보인다.
    - 실제 데이터들을 넣었을 때 효율적
    - 최악의 경우에도 성능을 유지한다.

## 추가 정리

- 공간복잡도
    
    공간복잡도: 특정 입력에 대해 알고리즘이 얼마나 많은 메모리를 차지하는가?
    
    - 빅오 표기법
        
        ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%206.png)
        
        ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%207.png)
        
- O(n^2)정렬
    - 버블 정렬
        - 설명
            
            인접한 두 원소를 비교해 조건에 맞지 않은 경우 자리를 교환해 정렬
            
            1. 인접한 2개의 값을 비교해 순서가 맞지 않는 경우 교환
            2. 교환이 더이상 이루어지지 않을 때까지 1번 반복
            - 예시
                
                ![bubble.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/bubble.gif)
                
        - 코드
            
            ```java
            public class Main {
            
            	public static void main(String[] args) throws IOException {
            		int[] num = {5, 3, 1, 2, 4};
            		
            		num = bubble_sort(num.length, num);
            		for (int k : num) {
            			System.out.print(k + " ");
            		}
            	}
            	
            	public static int[] bubble_sort(int n, int[] num) {
            		for (int i = n - 1; i > 0; --i) {
                        //정렬되지 않은 영역내의 모든 원소를 조건에 맞도록 교환한다.
            			for (int j = 0; j < i; ++j) {
            				if (num[j] > num[j + 1]) {
            					int temp = num[j];
            					num[j] = num[j + 1];
            					num[j + 1] = temp;
            				}
            			}
            		}
            		return num;
            	}
            }
            ```
            
        - 장점
            - 구현이 단순함
            - 추가 메모리 공간이 필요하지 않음
            - 안정 정렬
        - 단점
            - 시간 복잡도가 N^2
            - 정렬이 되어 있지 않는 원소를 교환하기 위해 원소의 교환이 다른 정렬에 비해 많이 발생
    - 선택 정렬
        - 설명
            
            원소를 넣을 위치를 미리 정해놓고 그 위치에 어떤 원소를 넣을지 선택해 정렬
            
            1. 배열의 최소 원소를 찾는다
            2. 그 원소를 맨 앞의 원소와 교체한다.
            3. 정렬된 부분 다음의 배열로 1번, 2번을 반복 수행한다.
            - 예시
                
                ![select.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/select.gif)
                
        - 코드
            
            ```java
            public class Main {
            
            	public static void main(String[] args) throws IOException {
            		int[] num = {5, 3, 1, 2, 4};
            		num = selection_Sort(num.length, num);
            		for (int k : num) {
            			System.out.print(k + " ");
            		}
            	}
            	
            	public static int[] selection_Sort(int n, int[] num) {
            		for (int i = 0; i < n - 1; ++i) {
            			int index = i;
            			for (int j = i + 1; j < n; ++j) {
                            //정렬되지 않은 영역에서 가장 작은 원소의 인덱스를 찾음
            				if (num[j] < num[index]) {
            					index = j;
            				}
            			}
            
            			int temp = num[i];
            			num[i] = num[index];
            			num[index] = temp;
            		}
            		return num;
            	}
            }
            ```
            
        - 장점
            - 구현하기 쉬움
            - 버블 정렬에 비해 교환의 횟수가 적음
            - 추가 메모리 공간이 필요없음.
        - 단점
            - 시간복잡도가 O(N^2)
            - 불안정 정렬
    - 삽입 정렬
        - 설명
            
            2번째 원소를 앞의 원소들과 비교해 삽입할 위치를 지정하고 다른 원소들을 옮긴 뒤 해당 자리에 원소를 삽입해 정렬
            
            1. 위치를 하나 잡고 해당 값을 temp에 저장
            2. temp와 해당 값의 앞에 있는 원소들과 비교해 삽입
            3. 1번에서 다음 위치를 정하고 2번을 반복한다.
            - 예시
                
                ![insert.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/insert.gif)
                
        - 코드
            
            ```java
            public class Main {
            
            	public static void main(String[] args) throws IOException {
            		int[] num = {5, 3, 1, 2, 4};
            		
            		num = insertion_sort(num.length, num);
            		for (int k : num) {
            			System.out.print(k + " ");
            		}
            	}
            	
            	public static int[] insertion_sort(int n, int[] num) {
            		for (int i = 1; i < n; ++i) {
            			int key = num[i];
            			int j = i - 1;
            			
            			//정렬된 배열은 0부터 i-1까지이므로 i-1부터 역순으로 조사한다.
            			//j >= 0이며, j의 원소가 Key보다 크다면 한칸 앞으로 옮겨준다.
            			for (; j >= 0 && num[j] > key ; --j) {
            				num[j + 1] = num[j];
            			}
            			
            			//for문을 탈출했다는 것은 j번째 원소가 Key보다 작다는 뜻이기 때문에 j + 1의 배열에 Key값을 넣어준다.
            			num[j + 1] = key;
            		}
            		return num;
            	}
            }
            ```
            
        - 장점
            - 구현이 쉬움
            - 이미 정렬된 경우 더 효율적
            - 추가 메모리 공간이 필요없음
            - 안정 정렬
            - 최선의 경우가 O(N)으로 다른 O(N^2)정렬에 비해 효율적
        - 단점
            - 최악의 경우에는 O(N^2)의 시간복잡도
- 기수 정렬
    - 설명
        
        데이터의 값이 동일한 길이를 가지는 숫자나 문자열로 구성되어 있는 경우에 사용하는 정렬
        
        - 예시
            
            ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%208.png)
            
            ![ridix.gif](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/ridix.gif)
            
    - 코드
        
        ```java
        import java.util.LinkedList;
        import java.util.Queue;
        
        public class Main {
        
        	static final int bucketSize = 10;
        	
        	public static void main(String[] args) {
        		int[] arr = {28, 93, 39, 81, 62, 72, 38, 26};
        		
        		radix_Sort(arr.length, arr);
        		
        		for (int i = 0; i < arr.length; ++i) {
        			System.out.print(arr[i] + " ");
        		}
        	}
        	
        	public static void radix_Sort(int n, int[] arr) {
        		//bucket 초기화
        		Queue<Integer>[] bucket = new LinkedList[bucketSize];
        		for (int i = 0; i < bucketSize; ++i) {
        			bucket[i] = new LinkedList<>();
        		}
        		
        		int factor = 1;
        		
        		//정렬할 자릿수의 크기 만큼 반복한다.
        		for (int d = 0; d < 2; ++d) {
        			for (int i = 0; i < n; ++i) {
        				bucket[(arr[i] / factor) % 10].add(arr[i]);
        			}
        			
        			for (int i = 0, j = 0; i < bucketSize; ++i) {
        				while (!bucket[i].isEmpty()) {
        					arr[j++] = bucket[i].poll();
        				}
        			}
        			
        			factor *= 10;
        		}
        	}
        }
        ```
        
    - 장점
        - O(dn)의 시간복잡도
        - 문자열도 정렬 가능
        - 안정 정렬
    - 단점
        - 특정한 경우에만 사용이 가능 / 데이터 타입의 제한(자릿수를 이용하기 때문)
        - 추가 메모리를 필요로 함
- 계수 정렬
    - 설명
        
        배열의 인덱스를 이용해 정렬
        
        조건 : 값은 양수(배열의 인덱스는 양수만 존재), 값의 범위가 너무 크지 않아야 함(메모리 영역을 너무 많이 할당하지 않기 위해)
        
        1. 배열의 원소들이 몇번 나왔는지 배열에 기록한다.
        2. 1번의 배열을 누적으로 하는 누적합배열을 만든다.
        3. 입력받은 배열과 누적합 배열을 이용해 정렬 (안정적으로 정렬을 위해 입력 배열의 뒤에서부터)
        - 예시
            
            ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%209.png)
            
            ![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20(SORT)%20f7b86ddf61e64da9b6417cc318010330/Untitled%2010.png)
            
    - 코드
        
        ```java
        import java.util.Arrays;
        import java.util.Collections;
        
        public class CountingSort {
            public static void main(String[] args) {
                Integer[] a = {1, 0, 3, 1, 3, 1};
        
                a = sort(a);
        
                System.out.println(Arrays.toString(a));
            }
        
            public static Integer[] sort(Integer[] a) {
                int max = Collections.max(Arrays.asList(a));
                Integer[] aux = new Integer[a.length];
                Integer[] c = new Integer[max+1];
                Arrays.fill(c, 0);
        
                // 각 원소 갯수 계산
                for (int i=0; i<a.length; i++) {
                    c[a[i]] += 1;
                }
        
                // 누적합 계산
                for (int i=1; i<c.length; i++) {
                    c[i] += c[i-1];
                }
        
                // 누적합을 이용해 정렬
                for (int i=a.length-1; i>=0; i--) {
                    aux[--c[a[i]]] = a[i];
                }
        
                return aux;
            }
        }
        ```
        
    - 장점
        - O(N)의 시간복잡도
    - 단점
        - 특정한 경우에만 사용가능
        - 추가적인 메모리 공간 필요 (분포가 큰 경우 메모리 낭비가 많을 수 있음)


https://ssafycsstudy.notion.site/SORT-952e4b5625fa4f4e967acde0aeecf9fa
