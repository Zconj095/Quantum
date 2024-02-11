using UnityEngine;
using System.Collections.Generic;

public class PhononPhotonInteractionSimulator : MonoBehaviour
{
    public GameObject photonPrefab; // Prefab representing the photon
    public List<GameObject> phononVertices = new List<GameObject>();
    public float interactionStrength = 0.1f; // Simplified representation of interaction strength

    private GameObject photon;

    void Start()
    {
        InitializePhoton();
        InitializePhononVertices();
    }

    void Update()
    {
        SimulateInteraction();
    }

    void InitializePhoton()
    {
        // Instantiate the photon object at a specific position with an explicit cast to GameObject
        photon = (GameObject)Instantiate(photonPrefab, new Vector3(0, 0, 0), Quaternion.identity);
    }


    void InitializePhononVertices()
    {
        // Initialize phonon vertices around the photon's initial position
        for (int i = 0; i < 5; i++) // Creating 5 phonon vertices as an example
        {
            GameObject phononVertex = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            phononVertex.transform.position = Random.insideUnitSphere * 5; // Position randomly within a radius
            phononVertices.Add(phononVertex);
        }
    }

    void SimulateInteraction()
    {
        // Move phonon vertices based on their interaction with the photon
        foreach (GameObject phononVertex in phononVertices)
        {
            Vector3 directionToPhoton = photon.transform.position - phononVertex.transform.position;
            directionToPhoton.Normalize();
            // Simulate phonon translation towards the photon as a simplified interaction effect
            phononVertex.transform.position += directionToPhoton * interactionStrength * Time.deltaTime;
        }
    }
}
