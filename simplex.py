import numpy as np
import random
import time
from scipy.optimize import linprog

class Simplex_Table(object):

    """ FUNC:
        get init simplex table
    """
    def __init__(self, ori_c, ori_A, ori_b, ori_res, var_type_arr, cons_type_arr, debug=False):  
        
        # defined constant   
        self.NINF   = -1000000
        self.INF    =  1000000
        self.NORMAL = -1

        # debug option
        self.debug  = debug

        # numpy for std simplex method
        self.std_c  = ori_c.copy()
        self.std_A  = ori_A.copy()
        self.std_b  = ori_b.copy()

        self.ori_constrain_num = ori_A.shape[0]
        self.ori_var_num       = ori_A.shape[1]
        self.var_type_arr      = var_type_arr
        self.cons_type_arr     = cons_type_arr  

        """ var_status array elements
            NORMAL means var x
            NINF   means var x is replaced by -x
            POS    means var x is made from a pair (x+, x-) 
        """
        self.var_status = np.full(self.ori_var_num,self.NORMAL)

        #change vars based on limitation types
        for i in range(len(var_type_arr)):
            # vars have <= 0 limitation
            if (var_type_arr[i] == -1):
                ori_A[:,i] = -ori_A[:,i]
                ori_c[:,i] = -ori_c[:,i]

                self.var_status[i] = self.NINF

            # vars have no limitation
            if (var_type_arr[i] == 0):
                col_A = np.array(-ori_A[:,i]).reshape(-1,1)
                col_c = np.array(-ori_c[:,i]).reshape(-1,1)
                ori_A = np.concatenate((ori_A, col_A),axis=1).reshape(self.ori_constrain_num,-1)
                ori_c = np.concatenate((ori_c, col_c),axis=1).reshape(1,-1)

                self.var_status[i] = i
                self.var_status    = np.append(self.var_status, i)

            # vars have >=0 limitation
            if (var_type_arr[i] == 1):
                pass
        
        """ add auxiliary vars based on constrain types """
        for i in range(len(cons_type_arr)):
            # constrains <= type
            if (cons_type_arr[i] == -1):
                col_A       = np.zeros(shape=(self.ori_constrain_num,1))
                col_A[i][0] = 1
                col_c       = np.zeros(shape=(1,1))  
                ori_A       = np.concatenate((ori_A,col_A), axis=1).reshape(self.ori_constrain_num,-1)
                ori_c       = np.concatenate((ori_c,col_c), axis=1).reshape(1,-1)

                self.var_status = np.append(self.var_status, self.NORMAL)
            
            # constrains = type
            if (cons_type_arr[i] == 0):
                pass

            # constrains >= type
            if (cons_type_arr[i] == 1):
                ori_A[i,:]  = -ori_A[i,:]
                b[i,:]      = -b[i,:]
                col_A       = np.zeros(shape=(self.ori_constrain_num,1))
                col_A[i][0] = 1
                col_c       = np.zeros(shape=(1,1))    
                ori_A       = np.concatenate((ori_A,col_A), axis=1).reshape(self.ori_constrain_num,-1)
                ori_c       = np.concatenate((ori_c,col_c), axis=1).reshape(1,-1)

                self.var_status = np.append(self.var_status, self.NORMAL)

        """ IMPORTANT STEP 
            if b[i] < 0, we need to get its negative version to avoid **dead-loop**
        """ 
        for i in range(ori_b.shape[0]):
            if (ori_b[i,:] < 0):
                ori_A[i,:] = -ori_A[i,:]
                ori_b[i,:] = -ori_b[i,:]
        

        """ init ingredients for simplex method """
        self.c             = ori_c
        self.A             = ori_A
        self.b             = ori_b
        self.res           = ori_res
        self.res_type      = 1
        self.constrain_num = (self.A).shape[0]
        self.var_num       = (self.A).shape[1]

        """ standard simplex form
            1. vars all >=0 type
            2. cons all ==  type
            3. b    all >=0 type
            4. auxiliary vars added
        """
        if (self.debug == True):
            print("\nSimplex Table Init...")
            self.print_simplex_table(
                                    [i for i in range(self.constrain_num)], 
                                    [i + self.constrain_num for i in range(self.var_num - self.constrain_num)],
                                    self.A, 
                                    np.empty(shape=(0,0)), 
                                    self.c, 
                                    np.empty(shape=(0,0)), 
                                    self.b, 
                                    self.res
                                    )

            print("Variable Status...")
            self.print_var_status(self.var_status)

    """ FUNC:
        update simplex table in new base
    """
    def simplex_loop(self,base_list, null_list, A_b, A_n, c_b, c_n, b, res):

        in_index_col_have_pos = True
        no_loop_base_change = True
        no_inf_solution = True
        in_index = -1
        out_index = -1

        while (1):
            # remember in/out index at last iteration
            prev_in_index = in_index
            prev_out_index = out_index

            # change simplex table
            A_b, A_n, c_b, c_n, b, res = self.transform(A_b, A_n, c_b, c_n, b, res)
            if (self.debug == True):
                print("After transforming...")
                self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
            
            # decide end of simplex loop 
            if (self.decide_end_simplex_or_not(c_n, b)):
                if (self.debug == True): print("Judging finish simplex or not... Finally End!")
                break
            else:
                if (self.debug == True): print("Judging finish simplex or not... Keep Going!")
            
            # change base
            base_list, null_list, A_b, A_n, c_b, c_n, in_index_col_have_pos, in_index, out_index = self.change_base(base_list, null_list, A_b, A_n, c_b, c_n, b)
            if (self.debug == True): self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

            # detect situations like a1-->a2  a2--->a1 a1--->a2 ...
            if (prev_in_index == out_index and prev_out_index == in_index):
                no_loop_base_change = False

            # detect inf solution return type
            if (in_index_col_have_pos == False or no_loop_base_change == False):
                no_inf_solution = False
                break

        return base_list, null_list, A_b, A_n, c_b, c_n, b, res, no_inf_solution

    """ FUNC:
        do 2-stage simplex method
    """
    def simplex_algo(self):
        
        # no solution type 1 (rank error check)
        if (self.check_rank(self.A, self.c) == -1):
            return -1,0,[]

        
        # 1st stage var
        base_list = [i+self.var_num for i in range(self.constrain_num)]
        null_list = [i for i in range(self.var_num)]
        A_n = self.A.copy()
        A_b = np.eye(self.constrain_num)
        c_n = np.zeros(shape=(1,self.var_num))
        c_b = np.full((1,self.constrain_num),-1)
        b   = self.b.copy()   
        res = 0

        if (self.debug == True):
            print("\n======1st Stage for Simplex Method======")
            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
        
        # begin do 1st stage simplex method
        base_list, null_list, A_b, A_n, c_b, c_n, b, res, no_inf_solution = self.simplex_loop(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

        # inf solution error
        if (no_inf_solution == False):
            return 0, 0, []

        if (self.debug == True):
            print("\nInitial point FOUND!")
            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

        # no solution type 2 (no init point check)
        # we can find situation that can not find init point
        if(abs(res-0) >= 1e-10):
            return -1, 0, []
        
        # update para in simplex method
        self.null_list = [i for i in null_list if i < self.var_num]
        self.base_list = [i for i in base_list if i < self.var_num]

        if (len(self.base_list) < self.constrain_num):
            self.base_list += self.null_list[-(self.constrain_num - len(self.base_list)):]
            self.null_list  = self.null_list[:-(self.constrain_num - len(self.base_list))]
        self.A_b, self.A_n  = self.split_matrix(self.A, self.base_list, self.null_list)
        self.c_b, self.c_n  = self.split_matrix(self.c, self.base_list, self.null_list)
        
        if (self.debug == True):
            print("\n======2nd Stage for Simplex Method======")
            self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)


        # begin do 2nd stage simplex method
        self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res, no_inf_solution = self.simplex_loop(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
        if (no_inf_solution == False):
            return 0, 0, []

        if (self.debug == True):
            print("\nComplete, get final results!")
            self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
        
        # combine and adjutst auxiliary vars
        self.res_type, self.opt_solution, self.solution = self.handle_result(self.res_type, self.base_list, self.null_list, (self.b).reshape(-1), self.res, self.var_status)

        return self.res_type, self.opt_solution, self.solution


    def std_simplex_algo(self):
        # prepare ingredients for std simplex API
        c = list(-self.std_c[0,:])
        A_ub = []
        A_eq = []
        b_ub = []
        b_eq = []
        for i in range(self.A.shape[0]): 
            if (self.cons_type_arr[i] == -1):
                A_ub.append(list(self.std_A[i]))
                b_ub.append(self.std_b[i][0])
            elif (self.cons_type_arr[i] == 0):
                A_eq.append(list(self.std_A[i]))
                b_eq.append(self.std_b[i][0])
            elif (self.cons_type_arr[i] == 1):
                A_ub.append(list(-self.std_A[i]))
                b_ub.append(-self.std_b[i][0])
        bound = []
        for i in range(len(self.var_type_arr)):
            if (self.var_type_arr[i] == 1):
                bound.append((0, None))
            if (self.var_type_arr[i] == -1):
                bound.append((None, 0))
            if (self.var_type_arr[i] == 0):
                bound.append((None, None))
        
        # use different paras
        if (A_ub == []):
            res = linprog(c, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=bound, method='simplex')
        elif (A_eq == []):
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=bound, method='simplex')
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bound, method='simplex')

        """res.status of API
           res.status 0 : normal
           res.status 1 : reach max iter number
           res.status 2 : no solution
           res.status 3 : inf solution
           res.status 4 : math error

           res type of PROJECT
           res type -1 : no solution
           res type 0  : inf solution
           res type 1  : normal
        """
        res_type = 1
        if (res.status == 2 or res.status == 4):
            res_type = -1
        if (res.status == 3):
            res_type = 0
        if (res.status == 0 or res.status == 1):
            res_type = 1

        self.res_type = res_type
        self.opt_solution = res.fun
        self.solution = res.x

        return self.res_type, self.opt_solution, self.solution


    def find_in_index(self, c_n, null_list):
        return null_list[np.argmax(c_n)]


    def find_out_index(self, A_n, b, in_index, null_list,base_list):
        in_index_pos_in_null_list = null_list.index(in_index)
        A_n_in_index_col          = A_n[:,in_index_pos_in_null_list].reshape(-1,1)
        A_n_in_index_col_copy     = A_n_in_index_col.copy()

        """ find min bi/ai,j (ai,j > 0 bi >= 0)
        """
        # since we might come across divide zero exception, we change all 0 in the np array into -1
        # we would not choose ai,j = 0 or -1, so changes make no effect
        A_n_in_index_col_copy[A_n_in_index_col_copy == 0] = -1
        div_mat = b / A_n_in_index_col_copy

        # if the min element in div_mat is negative
        # we replace min element to be self.INF and continue indexing argmin 
        while(A_n_in_index_col_copy[np.argmin(div_mat)] <= 1e-5):
            div_mat[np.argmin(div_mat),:] = np.array(self.INF).reshape(1,-1)
            if (np.min(div_mat) == self.INF):
                break
        
        # special situation : chosen in index all elements are neg
        # solutions be INF or -INF        
        in_index_col_have_pos = True
        if (np.min(div_mat) == self.INF):
            in_index_col_have_pos = False

        return base_list[np.argmin(div_mat)], in_index_col_have_pos


    def change_base(self, base_list, null_list, A_b, A_n, c_b, c_n, b):
        # get in_index, out_index
        in_index = self.find_in_index(c_n, null_list)
        out_index, in_index_col_have_pos = self.find_out_index(A_n, b, in_index, null_list, base_list)

        # find index in the list
        in_index_pos_in_null_list = null_list.index(in_index)
        out_index_pos_in_base_list = base_list.index(out_index)

        # swap two columns
        A_tmp = A_n[:, in_index_pos_in_null_list].copy()
        A_n[:, in_index_pos_in_null_list] = A_b[:, out_index_pos_in_base_list]
        A_b[:, out_index_pos_in_base_list] = A_tmp

        # swap tow columns
        c_tmp = c_n[:, in_index_pos_in_null_list].copy()
        c_n[:, in_index_pos_in_null_list] = c_b[:, out_index_pos_in_base_list]
        c_b[:, out_index_pos_in_base_list] = c_tmp

        # modify two lists
        null_list[in_index_pos_in_null_list] = out_index
        base_list[out_index_pos_in_base_list] = in_index

        if (self.debug == True):
            print("Changing base... IN={}, OUT={}".format(in_index, out_index))

        return base_list, null_list, A_b, A_n, c_b, c_n, in_index_col_have_pos, in_index, out_index


    def split_matrix(self,mat,lista, listb):
        return mat[:,lista], mat[:,listb]


    def inv_matrix(self,mat):
        return np.linalg.inv(mat)


    def transform(self, A_b, A_n, c_b, c_n, b, res):
        res = res - float(np.dot(np.dot(c_b, self.inv_matrix(A_b)) , b))
        c_n = c_n - np.dot(np.dot(c_b, self.inv_matrix(A_b)), A_n)
        A_n = np.dot(self.inv_matrix(A_b), A_n)
        b   = np.dot(self.inv_matrix(A_b), b)
        A_b = np.eye(A_b.shape[0])
        c_b = np.zeros(shape=(c_b.shape))
        return A_b, A_n, c_b, c_n, b, res
    

    def print_var_status(self, var_status):
        for i in range(len(self.var_status)):
            if (self.var_status[i] == self.NORMAL):
                print("x{}".format(i),end="  ")
            elif (self.var_status[i] == self.NINF):
                print("-x{}(x{})".format(i,i),end="  ")
            elif (self.var_status[i] == i):
                print("x{}+(x{})".format(i,i),end="  ")
            elif (self.var_status[i] >= 0):
                print("x{}-(x{})".format(self.var_status[i],i),end="  ")
        print("\n")


    def print_simplex_table(self, base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("------",end='')
        print("+",end='')
        print("\n",end='')

        # base & null index
        print("|",end="")
        for i in range(len(base_list)):
            print("%5d"%base_list[i],end=" ")
        for i in range(len(null_list)):
            print("%5d"%null_list[i],end=" ")
        print("      |\n",end="")

        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("------",end='')
        print("+",end='')
        print("\n",end='')
        
        # c & res array row
        print("|",end="")
        for i in range(c_b.shape[1]):
            print("%5.2f"%c_b[0][i],end=" ")
        for i in range(c_n.shape[1]):
            print("%5.2f"%c_n[0][i],end=" ")
        print("|%5.2f"%res,end="")
        print("|\n",end="")

        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("------",end='')
        print("+",end='')
        print("\n",end='')

        # A | b  matrix row
        for i in range(self.constrain_num):
            print("|",end="")
            for j in range(A_b.shape[1]):
                print("%5.2f"%A_b[i][j],end=" ")
            for j in range(A_n.shape[1]):
                print("%5.2f"%A_n[i][j],end=" ")
            print("|%5.2f"%b[i],end="")
            print("|\n",end="")
        
        # ===============
        print("+",end='')
        for i in range(len(base_list+null_list)+1):
            print("------",end='')
        print("+",end='')
        print("\n",end='')        


    def print_result(self):
        # 3 situation, print in diff forms
        if (self.res_type == 1):
            print(self.res_type)
            print(self.opt_solution)
            for i in range(len(self.solution)-1):
                print(self.solution[i],end=" ")
            print(self.solution[len(self.solution)-1])
        elif (self.res_type == 0):
            print(self.res_type)
        elif (self.res_type == -1):
            print(self.res_type)
    

    def handle_result(self, res_type, base_list, null_list, base_solution, opt_solution, var_status):
        # construct the solution list 
        solution = [0 for i in range(len(base_list+null_list))]
        for i in range(len(base_list)):
            solution[base_list[i]] = base_solution[i]

        # change solution list based on var_status
        for i in range(len(solution)):
            if (var_status[i] == self.NINF):
                solution[i] = -solution[i]
            elif (var_status[i] == self.NORMAL):
                continue
            elif (var_status[i] != i):
                solution[var_status[i]] -= solution[i]
        # selectt part of solution
        solution = solution[:self.ori_var_num]

        return res_type, opt_solution, solution


    def decide_end_simplex_or_not(self, c_n, b):
        # not 0 but -1e-5 ~ 1e-5
        # c_n might be empty, simplex shoulde be ended in one step
        if (c_n.shape == (1,0)):
            return True
        else:
            return True if ((np.max(c_n)<=1e-5 and np.min(b) >=-1e-5)) else False


    def check_rank(self, A, c):
        if (np.linalg.matrix_rank(A) != A.shape[0]):
            find_A_not_full_row_rank_res = True  
        else:
            find_A_not_full_row_rank_res = False 

        if (find_A_not_full_row_rank_res == True):
            return -1
        else: 
            return 1

    




if __name__ == '__main__':
    
    """ control program mode """
    test = True

    if (test == False):

        """ normal mode
            input required data
            output result of two algo
        """

        # 1st line of input
        # variable number and constrain number
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != 2 or int(var_constrain_str_split[0]) <=0 or int(var_constrain_str_split[1]) <= 0): 
            raise Exception('Variable and constrain number input ERROR!')
        var_num = int(var_constrain_str_split[0])
        constrain_num = int(var_constrain_str_split[1])

        # 2nd line of input
        # paramters for target function ( min -> by default)
        target_func_para_str = input()
        target_func_para_str_split = target_func_para_str.split()
        if (len(target_func_para_str_split) != var_num): 
            raise Exception('Parameters for target function input ERROR!')
        # change the (min) opt problem into (max) opt problem
        target_func_para_arr = [-int(i) for i in target_func_para_str_split]

        # init for important numpy variable
        # c must be a row vector
        c = np.array(target_func_para_arr).reshape(1,-1)
        # A,b must be empty numpy array
        A = np.empty(shape=(0, var_num))
        b = np.empty(shape=(0, 1))
        # res must be zero scalar
        res = 0

        # help the construction auxiliary variables
        constrain_equality = []
        # (constrain_number) lines lf input
        for i in range(constrain_num):
            constrain_str = input()
            constrain_str_split = constrain_str.split()
            if (len(constrain_str_split) != (var_num+2)): 
                raise Exception('input for constrain ERROR! constrain_str length = {}'.format(len(constrain_str.split())))

            constrain_para_arr = [int(i) for i in constrain_str_split]
            if (constrain_para_arr[-1] != -1 and constrain_para_arr[-1] != 0 and constrain_para_arr[-1] != 1): 
                raise Exception('input for constrain ERROR! last number in the constraint must be -1 or 1 or 0')

            # modify the constrain into standard forms ax+by <= c if it is ax+by >= c
            if (constrain_para_arr[-1] == 1):
                constrain_para_arr = [-i for i in constrain_para_arr]

            constrain_equality.append(constrain_para_arr[-1])

            row_A = np.array(constrain_para_arr[:-2]).reshape(-1,var_num)
            row_b = np.array(constrain_para_arr[-2]).reshape(-1,1)
            A = np.concatenate((A, row_A),axis=0).reshape(-1, var_num)
            b = np.concatenate((b, row_b),axis=0).reshape(-1, 1)

        # one lines for variable limitation
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != var_num):
            raise Exception('input for constraints on variables ERROR!')

        var_constrain_arr = [int(i) for i in var_constrain_str_split]
        
        simplex = Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality,debug=True)
        res_type, opt_solution, solution = simplex.simplex_algo()
        simplex.print_result()
        res_type, opt_solution, solution = simplex.std_simplex_algo()
        simplex.print_result()

    elif (test == True):

        """ test mode
            use generated random data
            use both std and self-build simplex-algo
            check error within 1e-5
        """
        constrain_num = 4
        variable_num = 5


        iter_num = 0
        minus_one_res = 0
        zero_res = 0
        one_res  = 0
        while(1):
            iter_num += 1

            # generate random init data
            np.random.seed(random.randint(0,12345)+iter_num % 12345)
            c = np.random.randn(1,variable_num)
            A = np.random.randn(constrain_num, variable_num)
            while (np.linalg.matrix_rank(A) != constrain_num):
                A = np.random.randn(constrain_num, variable_num)
            b = np.random.randn(constrain_num, 1)
            res = 0
            var_constrain_arr = []
            constrain_equality = []
            for i in range(variable_num):
                var_constrain_arr.append(np.random.randint(-1,2))
            for i in range(constrain_num):
                constrain_equality.append(np.random.randint(-1,2))

            # generate solution from two methods
            simplex = Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality,debug=False)
            std_res_type, std_opt_solution, std_solution = simplex.std_simplex_algo()
            #if (std_res_type != 1):
            #    continue
            res_type, opt_solution, solution = simplex.simplex_algo()

            # check the same or no 
            if (res_type == -1):
                minus_one_res += 1
            if (res_type == 0):
                zero_res += 1
            if (res_type == 1):
                one_res  += 1

            if (res_type != std_res_type):
                raise Exception('answer type error')
            if (res_type == 1 and abs(opt_solution - std_opt_solution) > 1e-5):
                raise Exception('optimal solution error')
            if (res_type == 1):
                var_check = True
                for i in range(len(solution)):
                    if (abs(solution[i] - std_solution[i]) > 1e-5):
                        var_check = False
                        break
                if (var_check == False):
                    raise Exception('variable solution error')
            print("\r {} data PASS. sum {} -1-type, {} 0-type, {} 1-type".format(iter_num, minus_one_res, zero_res, one_res),end='')
        

