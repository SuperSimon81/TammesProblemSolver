
import numpy as np
from numpy import sin,cos,sqrt
import random

def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
                    [ 0, cos(theta),-sin(theta)],
                    [ 0, sin(theta), cos(theta)]],dtype=np.float32)

def Ry(theta):
    return np.array([[ cos(theta), 0, sin(theta)],
                    [ 0           , 1, 0           ],
                    [sin(theta), 0, cos(theta)]],dtype=np.float32)

def Rz(theta):
    return np.array([[ cos(theta), -sin(theta), 0 ],
                     [ sin(theta), cos(theta) , 0 ],
                     [ 0           , 0            , 1 ]],dtype=np.float32)


def cart2sph(x,y,z):
    
    r = np.sqrt(x*x+y*y+z*z)
    phi = np.arctan2(y,x) 
    theta = np.arccos(z/r)  
    return r,phi,theta

def sph2cart(r,phi,theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
#vec3<f32>(r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta));

def circle_on_sphere(r,beta,gamma,t):
    x = (sin(r)*cos(beta)*cos(gamma))*cos(t)-sin(r)*sin(gamma)*sin(t)+(cos(r)*sin(beta)*cos(gamma))
    y = (sin(r)*cos(beta)*sin(gamma))*cos(t)+(sin(r)*cos(gamma))*sin(t)+(cos(r)*sin(beta)*sin(gamma))
    z = -(sin(r)*sin(beta))*cos(t)+cos(r)*cos(beta)
    return x,y,z

class Circle():
    
    def __init__(self,radius,theta,phi,nr_vertices):
        self.radius = radius
        self.theta = theta
        self.phi = phi
        self.nr_vertices = nr_vertices
        self.neighbours = []

    def update_circle(self,radius,theta,phi):
        radius = self.radius
        theta = self.theta
        phi = self.phi

    def get_circle(self,offset):
        radius = self.radius
        theta = self.theta
        phi = self.phi
        nr_of_vertices = self.nr_vertices
        
        lis = []
        for i in np.linspace(0,2*np.pi,nr_of_vertices):
            lis.append(np.array([1,  i,radius],dtype=np.float32))
        sph_circle = np.array(lis)
        cartesian = np.array([sph2cart(x,y,z) for x,y,z in sph_circle])
        rotated = np.array([np.asarray([x,y,z]@Rx(phi)@Rz(theta)).squeeze() for x,y,z in cartesian])
        vertices = np.array([cart2sph(x,y,z) for x,y,z in rotated])
        indices = []
        for i in range(0,len(vertices)-1):
            indices.append(i+offset)
            indices.append(i+1+offset)
        indices = np.array(indices,dtype=np.uint32).flatten() 
        self.vertices = vertices
        self.indices = indices
        return vertices, indices

    def get_tri_circle(self,offset):
        radius = self.radius
        theta = self.theta
        phi = self.phi

        lis = []
        #lis.append(np.array([0.001,  0,0],dtype=np.float32))
        lis.append(np.array([1,  0,0],dtype=np.float32))
        for i in np.linspace(0,2*np.pi,self.nr_vertices):
            lis.append(np.array([1,  i,radius],dtype=np.float32))
        sph_circle = np.array(lis)
        cartesian = np.array([sph2cart(x,y,z) for x,y,z in sph_circle])
        rotated = np.array([np.asarray([x,y,z]@Rx(phi)@Rz(theta)).squeeze() for x,y,z in cartesian])
        vertices = np.array([cart2sph(x,y,z) for x,y,z in rotated])
        indices = []
        
        for i in range(1,len(vertices)-1):
            indices.append(0+offset)
            indices.append(i+offset)
            indices.append(i+1+offset)
            
        indices = np.array(indices,dtype=np.uint32).flatten() 
        
        self.vertices = vertices
        self.indices = indices
        return vertices, indices
    def sphere_dist(self,other):

        return 2*np.arcsin(self.r3_euclid_dist(other)/2)
    
    def r3_euclid_dist(self,other):
        p0 = sph2cart(1,self.theta,self.phi)
        p1 = sph2cart(1,other.theta,other.phi)

        return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)

class Sphere(): 
    def __init__(self,num_pts):
        self.num_pts = num_pts
        self.radius = 0.2
        self.points = []
        verts = []
        inds = []
        self.steps = 0
        #self.triangles(self.radius,self.num_pts,32)
        #self.lines(self.radius,self.num_pts,32)
        for i in range(0,num_pts):
            
            theta = np.pi * (1 + 5**0.5) * (i+0.5)
            phi = np.arccos(1 - 2*(i+0.5)/num_pts)
            self.points.append(Circle(0.2,theta, phi,32))
            vertices,indices = self.points[i].get_circle(32*i)
            verts.extend(vertices)
            inds.extend(indices)
        self.indices = np.array(inds).flatten()
        self.vertices = np.array(verts).flatten()

    def step(self):
        self.steps = self.steps + 1
        tot_list= self.check_distances()
        #print(self.points[ind].theta)
        ind1 = np.argmin(tot_list)
        ind = random.randint(0,len(self.points)-1)
        old_theta = self.points[ind].theta
        old_phi = self.points[ind].phi
        
        self.points[ind].theta=self.points[ind].theta+(random.random()*2-1)*self.points[ind].theta/1000
        self.points[ind].phi = self.points[ind].phi+(random.random()*2-1)*self.points[ind].phi/1000

        tot_list2 = self.check_distances()
      
        ind2 = np.argmin(tot_list2)
        val = tot_list[ind1]
        val2 = tot_list2[ind2]
        
        #print(f"{val-val2}")
       # if np.std(tot_list2[2]) < np.std(tot_list[2]):
            #print(f"{ np.std(tot_list2[2])}")
            #print("updated")
        #median = np.median(tot_list[2])
        #q75, q25 = np.percentile(tot_list[2], [75 ,25])
        test = tot_list2
        a = np.min(test)
        b = np.max(test)
        if val2>val:
            print(f"{a} {b} ")
        else:
            #print("did nothing")
            self.points[ind].theta = old_theta
            self.points[ind].phi = old_phi
        self.update()

    def check_distances(self):
        total_dist_list =[] 
        for i,a in enumerate(self.points):
            dist_list = []
            test = []
            for ii,b in enumerate(self.points):
                if a is not b:
                    dist_list.append(a.sphere_dist(b))
            dist_array = np.array(dist_list)
            smallest_neighbor = np.argmin(dist_array)
            #a.distances = []
            #a.distances.append( dist_list[a.neighbors[0]][2])
            #a.distances.append( dist_list[a.neighbors[1]][2])
            #a.distances.append( dist_list[a.neighbors[2]][2])

            a.radius = dist_array[smallest_neighbor]/2
            total_dist_list.append(dist_array[smallest_neighbor])
            #print(dist_list[1])
        #ind = np.argmax(total_dist_list[2])
        #ind2 = np.argmin(total_dist_list[2])
        #val = total_dist_list[ind]
        #print(f"{val} {total_dist_list[ind2]}")
        
        return np.array(total_dist_list)

    def update(self):
        verts = []
        inds = []
        for i,circle in enumerate(self.points):
            vertices,indices = circle.get_circle(32*i)
            verts.extend(vertices)
            inds.extend(indices)
        self.indices = np.array(inds).flatten()
        self.vertices = np.array(verts).flatten()
        

    def triangles(self,radius,num_pts,circle_vertices):
        verts = []
        inds = []
        for i in range(0,num_pts):
            phi = np.arccos(1 - 2*(i+0.5)/num_pts)
            theta = np.pi * (1 + 5**0.5) * (i+0.5)
            vertices, indices = self.put_tri_circle(theta,phi,radius,circle_vertices,i*(circle_vertices+1))
            verts.append(vertices)
            inds.append(indices)
        pass
        indices = np.array(inds).flatten()
        vertices = np.array(verts).flatten()

        self.indices = indices
        self.vertices = vertices

    def put_tri_circle(self,theta,phi,radius,nr_of_vertices,offset):
        lis = []
        #lis.append(np.array([0.001,  0,0],dtype=np.float32))
        lis.append(np.array([1,  0,0],dtype=np.float32))
        for i in np.linspace(0,2*np.pi,nr_of_vertices):
            lis.append(np.array([1,  i,radius],dtype=np.float32))
        sph_circle = np.array(lis)
        cartesian = np.array([sph2cart(x,y,z) for x,y,z in sph_circle])
        rotated = np.array([np.asarray([x,y,z]@Rx(phi)@Rz(theta)).squeeze() for x,y,z in cartesian])
        spherical = np.array([cart2sph(x,y,z) for x,y,z in rotated])
        indices = []
        
        
        for i in range(1,len(spherical)-1):
            indices.append(0+offset)
            indices.append(i+offset)
            indices.append(i+1+offset)
            
        indices = np.array(indices,dtype=np.uint32).flatten() 
        return spherical, indices

    def lines(self,radius,num_pts,circle_vertices):
        verts = []
        inds = []
        for i in range(0,num_pts):
            phi = np.arccos(1 - 2*(i+0.5)/num_pts)
            theta = np.pi * (1 + 5**0.5) * (i+0.5)
            vertices, indices = self.put_circle(theta,phi,radius,circle_vertices,i*circle_vertices)
            verts.append(vertices)
            inds.append(indices)
        pass
        indices = np.array(inds).flatten()
        vertices = np.array(verts).flatten()

        self.indices = indices
        self.vertices = vertices

    def put_circle(self,theta,phi,radius,nr_of_vertices,offset):
        lis = []
        for i in np.linspace(0,2*np.pi,nr_of_vertices):
            lis.append(np.array([1,  i,radius],dtype=np.float32))
        sph_circle = np.array(lis)
        cartesian = np.array([sph2cart(x,y,z) for x,y,z in sph_circle])
        rotated = np.array([np.asarray([x,y,z]@Rx(phi)@Rz(theta)).squeeze() for x,y,z in cartesian])
        spherical = np.array([cart2sph(x,y,z) for x,y,z in rotated])
        indices = []
        for i in range(0,len(spherical)-1):
            indices.append(i+offset)
            indices.append(i+1+offset)
        indices = np.array(indices,dtype=np.uint32).flatten() 
        return spherical, indices


    