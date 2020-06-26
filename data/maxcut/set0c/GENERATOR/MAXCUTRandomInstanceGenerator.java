import java.io.*;
import java.util.Random;

public class MAXCUTRandomInstanceGenerator
{
  public static int generateEdgeWeight( Random random )
  {
    int    result;
    double z1, z2;

    z1     = -Math.log(random.nextDouble());
    z2     = -Math.log(random.nextDouble());
    result = (int) ((z1/(z1+z2))*5)+1;
    result = result >= 6 ? 5 : result;

    return( result );
  }

  public static void main( String ps[] )
  {
    int    i, j, number_of_vertices, number_of_edges, weight, sqrt_number_of_vertices;
    Random random;

    random = new Random();

    number_of_vertices      = Integer.valueOf( ps[0] ).intValue();
    sqrt_number_of_vertices = (int) Math.sqrt( number_of_vertices );
    number_of_edges         = number_of_vertices + (sqrt_number_of_vertices-1)*(sqrt_number_of_vertices-1)-1 + 2*sqrt_number_of_vertices;

    System.out.println(number_of_vertices + " " + number_of_edges);
    /* Grid */
    for( i = 1; i <= number_of_vertices; i++ )
    {
      j = i+1;
      if( (i % sqrt_number_of_vertices) != 0 )
      {
        weight = generateEdgeWeight( random );
        System.out.println(i + " " + j + " " + weight);
      }

      j = i+sqrt_number_of_vertices;
      if( j <= number_of_vertices )
      {
        weight = generateEdgeWeight( random );
        System.out.println(i + " " + j + " " + weight);
      }
    }

    /* Extra edges to make it a torus */
    for( i = 1; i <= sqrt_number_of_vertices; i++ )
    {
      j      = number_of_vertices-sqrt_number_of_vertices+i;
      weight = generateEdgeWeight( random );
      System.out.println(i + " " + j + " " + weight);
    }
    for( i = 1; i <= number_of_vertices; i += sqrt_number_of_vertices )
    {
      j      = i+sqrt_number_of_vertices-1;
      weight = generateEdgeWeight( random );
      System.out.println(i + " " + j + " " + weight);
    }
  }
}
