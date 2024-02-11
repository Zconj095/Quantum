using UnityEngine;
using System.Collections.Generic;

public class AdvancedPhononSimulation : MonoBehaviour
{
    public int numberOfVertices = 20;
    public float baseTemperature = 300f; // Kelvin
    public float currentTemperature;
    public float thermalExpansionCoefficient = 0.00001f; // Simplified linear expansion coefficient

    private List<GameObject> phononVertices = new List<GameObject>();
    private List<Vector3> initialPositions = new List<Vector3>();

    void Start()
    {
        currentTemperature = baseTemperature;
        GeneratePhononVertices();
    }

    void Update()
    {
        // Example to simulate temperature change over time or based on an event
        if (Input.GetKeyDown(KeyCode.T)) // Press T to increase temperature
        {
            currentTemperature += 10f; // Increase temperature by 10K
            UpdatePhononVertexPositions();
        }
    }

    void GeneratePhononVertices()
    {
        for (int i = 0; i < numberOfVertices; i++)
        {
            GameObject vertex = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Vector3 randomPosition = Random.insideUnitSphere * 5; // Randomly position within a sphere
            vertex.transform.position = randomPosition;
            initialPositions.Add(randomPosition);
            phononVertices.Add(vertex);
        }
    }

    void UpdatePhononVertexPositions()
    {
        float temperatureDelta = currentTemperature - baseTemperature;
        for (int i = 0; i < phononVertices.Count; i++)
        {
            // Calculate new position based on original position, temperature change, and expansion coefficient
            Vector3 newPosition = initialPositions[i] * (1 + thermalExpansionCoefficient * temperatureDelta);
            phononVertices[i].transform.position = newPosition;
        }
    }
}
